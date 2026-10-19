"""Stage 2 models: M3 (M2 + process supervision), M3-flat, M3-shuffle.

Process labels: recall_hit_M, recall_hit_T, rerank_rescue_M, rerank_harm_M,
                rerank_rescue_T, rerank_harm_T (all binary).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from train.config import TEMPERATURE
from stage_models import BM_I, TD_I, TR_I, MD_I, MR_I


class M3Model(nn.Module):
    """M2 + process supervision. Stage1: recall_hit; Stage2: condition on recall to predict rescue/harm."""
    def __init__(self, backbone, hidden_size, alpha=1.0, beta=1.0, gamma=0.5):
        super().__init__()
        self.backbone = backbone
        self.stage1 = nn.Linear(hidden_size, 3)  # BM, TD, MD base utilities
        self.stage2 = nn.Linear(hidden_size + 3, 2)  # Δ_T, Δ_M
        # Process heads
        self.recall_head = nn.Linear(hidden_size, 2)  # recall_hit_M, recall_hit_T
        self.rerank_head = nn.Linear(hidden_size + 2, 4)  # rescue_M, harm_M, rescue_T, harm_T (condition on recall)
        nn.init.normal_(self.stage1.weight, std=0.02); nn.init.zeros_(self.stage1.bias)
        nn.init.normal_(self.stage2.weight, std=0.02); nn.init.zeros_(self.stage2.bias)
        nn.init.normal_(self.recall_head.weight, std=0.02); nn.init.zeros_(self.recall_head.bias)
        nn.init.normal_(self.rerank_head.weight, std=0.02); nn.init.zeros_(self.rerank_head.bias)
        self.alpha = alpha; self.beta = beta; self.gamma = gamma

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask)
        base_pred = self.stage1(h)
        stage2_in = torch.cat([h, base_pred], dim=-1)
        delta_pred = self.stage2(stage2_in)
        full_util = torch.zeros(h.size(0), 5, device=h.device)
        full_util[:, BM_I] = base_pred[:, 0]
        full_util[:, TD_I] = base_pred[:, 1]
        full_util[:, MD_I] = base_pred[:, 2]
        full_util[:, TR_I] = base_pred[:, 1] + delta_pred[:, 0]
        full_util[:, MR_I] = base_pred[:, 2] + delta_pred[:, 1]
        # Process predictions
        recall_logits = self.recall_head(h)
        rerank_in = torch.cat([h, torch.sigmoid(recall_logits)], dim=-1)
        rerank_logits = self.rerank_head(rerank_in)
        return {"base_pred": base_pred, "delta_pred": delta_pred, "util_pred": full_util,
                "logits": full_util, "recall_logits": recall_logits, "rerank_logits": rerank_logits}

    def loss(self, base_pred, delta_pred, util_pred, recall_logits, rerank_logits,
             reward_vec, ndcg_vec, proc_labels):
        # M2 loss
        target = F.softmax(reward_vec / TEMPERATURE, dim=-1)
        log_probs = F.log_softmax(util_pred / TEMPERATURE, dim=-1)
        per_sample = -(target * log_probs).sum(-1)
        mask = (reward_vec.max(-1)[0] > 0).float()
        kl = (per_sample * mask).sum() / mask.sum().clamp(min=1)
        base_true = ndcg_vec[:, [BM_I, TD_I, MD_I]]
        huber_base = F.huber_loss(base_pred, base_true, reduction="mean")
        delta_true = torch.stack([ndcg_vec[:, TR_I] - ndcg_vec[:, TD_I],
                                  ndcg_vec[:, MR_I] - ndcg_vec[:, MD_I]], dim=-1)
        huber_delta = F.huber_loss(delta_pred, delta_true, reduction="mean")
        # Process supervision loss (BCE)
        recall_labels = proc_labels[:, :2]  # recall_hit_M, recall_hit_T
        rerank_labels = proc_labels[:, 2:]  # rescue_M, harm_M, rescue_T, harm_T
        proc_loss = F.binary_cross_entropy_with_logits(recall_logits, recall_labels)
        proc_loss += F.binary_cross_entropy_with_logits(rerank_logits, rerank_labels)
        return kl + self.alpha * huber_base + self.beta * huber_delta + self.gamma * proc_loss


class M3FlatModel(nn.Module):
    """M3 but process heads are parallel (no recall->rerank dependency)."""
    def __init__(self, backbone, hidden_size, alpha=1.0, beta=1.0, gamma=0.5):
        super().__init__()
        self.backbone = backbone
        self.stage1 = nn.Linear(hidden_size, 3)
        self.stage2 = nn.Linear(hidden_size + 3, 2)
        self.recall_head = nn.Linear(hidden_size, 2)
        self.rerank_head = nn.Linear(hidden_size, 4)  # independent
        for m in [self.stage1, self.stage2, self.recall_head, self.rerank_head]:
            nn.init.normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)
        self.alpha = alpha; self.beta = beta; self.gamma = gamma

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask)
        base_pred = self.stage1(h)
        delta_pred = self.stage2(torch.cat([h, base_pred], dim=-1))
        full_util = torch.zeros(h.size(0), 5, device=h.device)
        full_util[:, BM_I] = base_pred[:, 0]
        full_util[:, TD_I] = base_pred[:, 1]
        full_util[:, MD_I] = base_pred[:, 2]
        full_util[:, TR_I] = base_pred[:, 1] + delta_pred[:, 0]
        full_util[:, MR_I] = base_pred[:, 2] + delta_pred[:, 1]
        recall_logits = self.recall_head(h)
        rerank_logits = self.rerank_head(h)
        return {"base_pred": base_pred, "delta_pred": delta_pred, "util_pred": full_util,
                "logits": full_util, "recall_logits": recall_logits, "rerank_logits": rerank_logits}

    def loss(self, base_pred, delta_pred, util_pred, recall_logits, rerank_logits,
             reward_vec, ndcg_vec, proc_labels):
        target = F.softmax(reward_vec / TEMPERATURE, dim=-1)
        log_probs = F.log_softmax(util_pred / TEMPERATURE, dim=-1)
        per_sample = -(target * log_probs).sum(-1)
        mask = (reward_vec.max(-1)[0] > 0).float()
        kl = (per_sample * mask).sum() / mask.sum().clamp(min=1)
        base_true = ndcg_vec[:, [BM_I, TD_I, MD_I]]
        huber_base = F.huber_loss(base_pred, base_true, reduction="mean")
        delta_true = torch.stack([ndcg_vec[:, TR_I] - ndcg_vec[:, TD_I],
                                  ndcg_vec[:, MR_I] - ndcg_vec[:, MD_I]], dim=-1)
        huber_delta = F.huber_loss(delta_pred, delta_true, reduction="mean")
        recall_labels = proc_labels[:, :2]
        rerank_labels = proc_labels[:, 2:]
        proc_loss = F.binary_cross_entropy_with_logits(recall_logits, recall_labels)
        proc_loss += F.binary_cross_entropy_with_logits(rerank_logits, rerank_labels)
        return kl + self.alpha * huber_base + self.beta * huber_delta + self.gamma * proc_loss
