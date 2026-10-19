"""Stage 1 models: M0 (KL only), M1 (KL + utility regression), M2 (stage-latent).

All share Qwen3-0.6B + LoRA backbone. Difference is only in heads and loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from train.config import LAMBDA, TEMPERATURE, ARM_NAMES, NDCG_COLS, LATENCY_COLS

ARM_IDX = {a: i for i, a in enumerate(ARM_NAMES)}  # MR=0, MD=1, TR=2, TD=3, BM=4
BM_I = ARM_IDX["BM25"]; TD_I = ARM_IDX["TEXT-SINGLE"]
TR_I = ARM_IDX["TEXT_RERANK"]; MD_I = ARM_IDX["MULTIMODAL-SINGLE"]
MR_I = ARM_IDX["MULTIMODAL_RERANK"]


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


class StageBackbone(nn.Module):
    """Qwen3 + LoRA + mean pool. Shared across M0/M1/M2."""
    def __init__(self, base_model, hidden_size, dropout=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        h = mean_pool(out.hidden_states[-1], attention_mask)
        return self.dropout(h)


class M0Model(nn.Module):
    """Original: single 5-way linear head, KL loss only."""
    def __init__(self, backbone, hidden_size, num_labels=5):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask)
        return {"logits": self.classifier(h)}

    def loss(self, logits, reward_vec):
        target = F.softmax(reward_vec / TEMPERATURE, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        per_sample = -(target * log_probs).sum(-1)
        mask = (reward_vec.max(-1)[0] > 0).float()
        return (per_sample * mask).sum() / mask.sum().clamp(min=1)


class M1Model(nn.Module):
    """KL + 5-arm utility regression (Huber)."""
    def __init__(self, backbone, hidden_size, num_labels=5, alpha=1.0):
        super().__init__()
        self.backbone = backbone
        self.reg_head = nn.Linear(hidden_size, num_labels)
        nn.init.normal_(self.reg_head.weight, std=0.02)
        nn.init.zeros_(self.reg_head.bias)
        self.alpha = alpha

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask)
        util_pred = self.reg_head(h)  # predicted nDCG utilities
        return {"util_pred": util_pred, "logits": util_pred}

    def loss(self, util_pred, reward_vec, ndcg_vec):
        # KL on reward-derived logits
        target = F.softmax(reward_vec / TEMPERATURE, dim=-1)
        log_probs = F.log_softmax(util_pred / TEMPERATURE, dim=-1)
        per_sample = -(target * log_probs).sum(-1)
        mask = (reward_vec.max(-1)[0] > 0).float()
        kl = (per_sample * mask).sum() / mask.sum().clamp(min=1)
        # Huber regression on nDCG utilities
        huber = F.huber_loss(util_pred, ndcg_vec, reduction="mean")
        return kl + self.alpha * huber


class M2Model(nn.Module):
    """Stage-latent: Stage1 predicts BM/TD/MD base util; Stage2 conditions on Stage1 to predict Δ_T, Δ_M."""
    def __init__(self, backbone, hidden_size, alpha=1.0, beta=1.0):
        super().__init__()
        self.backbone = backbone
        # Stage 1: predict base utilities (BM, TD, MD)
        self.stage1 = nn.Linear(hidden_size, 3)
        nn.init.normal_(self.stage1.weight, std=0.02)
        nn.init.zeros_(self.stage1.bias)
        # Stage 2: predict Δ_T, Δ_M conditioned on (h, stage1_pred)
        self.stage2 = nn.Linear(hidden_size + 3, 2)
        nn.init.normal_(self.stage2.weight, std=0.02)
        nn.init.zeros_(self.stage2.bias)
        self.alpha = alpha
        self.beta = beta

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask)
        base_pred = self.stage1(h)  # [BM, TD, MD]
        stage2_in = torch.cat([h, base_pred], dim=-1)
        delta_pred = self.stage2(stage2_in)  # [Δ_T, Δ_M]
        # Reconstruct full utility: [BM, TD, TD+Δ_T, MD, MD+Δ_M]
        full_util = torch.zeros(h.size(0), 5, device=h.device)
        full_util[:, BM_I] = base_pred[:, 0]
        full_util[:, TD_I] = base_pred[:, 1]
        full_util[:, MD_I] = base_pred[:, 2]
        full_util[:, TR_I] = base_pred[:, 1] + delta_pred[:, 0]
        full_util[:, MR_I] = base_pred[:, 2] + delta_pred[:, 1]
        return {"base_pred": base_pred, "delta_pred": delta_pred,
                "util_pred": full_util, "logits": full_util}

    def loss(self, base_pred, delta_pred, util_pred, reward_vec, ndcg_vec, latencies):
        # KL on full utility
        target = F.softmax(reward_vec / TEMPERATURE, dim=-1)
        log_probs = F.log_softmax(util_pred / TEMPERATURE, dim=-1)
        per_sample = -(target * log_probs).sum(-1)
        mask = (reward_vec.max(-1)[0] > 0).float()
        kl = (per_sample * mask).sum() / mask.sum().clamp(min=1)
        # Huber on base utilities (BM, TD, MD)
        base_true = ndcg_vec[:, [BM_I, TD_I, MD_I]]
        huber_base = F.huber_loss(base_pred, base_true, reduction="mean")
        # Huber on deltas
        delta_true = torch.stack([
            ndcg_vec[:, TR_I] - ndcg_vec[:, TD_I],
            ndcg_vec[:, MR_I] - ndcg_vec[:, MD_I],
        ], dim=-1)
        huber_delta = F.huber_loss(delta_pred, delta_true, reduction="mean")
        return kl + self.alpha * huber_base + self.beta * huber_delta


def make_backbone(base_model, hidden_size):
    return StageBackbone(base_model, hidden_size)


def compute_reward(ndcg, latencies):
    """reward = (1-lambda)*ndcg + lambda*(1 - latency/max_latency)."""
    lat = latencies / latencies.max()
    return (1 - LAMBDA) * ndcg + LAMBDA * (1 - lat)
