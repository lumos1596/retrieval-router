"""M2 ablation: remove stage1->stage2 dependency. Two independent parallel heads.

Same supervision (base utilities + deltas) but no conditional dependency.
Tests whether M2's gain comes from structure, not just extra params/labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from train.config import TEMPERATURE
from stage_models import BM_I, TD_I, TR_I, MD_I, MR_I


class M2ParallelModel(nn.Module):
    """Parallel heads: base head and delta head are independent (no stage1->stage2 link)."""
    def __init__(self, backbone, hidden_size, alpha=1.0, beta=1.0):
        super().__init__()
        self.backbone = backbone
        self.base_head = nn.Linear(hidden_size, 3)  # BM, TD, MD
        self.delta_head = nn.Linear(hidden_size, 2)  # Δ_T, Δ_M
        nn.init.normal_(self.base_head.weight, std=0.02); nn.init.zeros_(self.base_head.bias)
        nn.init.normal_(self.delta_head.weight, std=0.02); nn.init.zeros_(self.delta_head.bias)
        self.alpha = alpha; self.beta = beta

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask)
        base_pred = self.base_head(h)
        delta_pred = self.delta_head(h)  # independent, no conditioning on base_pred
        full_util = torch.zeros(h.size(0), 5, device=h.device)
        full_util[:, BM_I] = base_pred[:, 0]
        full_util[:, TD_I] = base_pred[:, 1]
        full_util[:, MD_I] = base_pred[:, 2]
        full_util[:, TR_I] = base_pred[:, 1] + delta_pred[:, 0]
        full_util[:, MR_I] = base_pred[:, 2] + delta_pred[:, 1]
        return {"base_pred": base_pred, "delta_pred": delta_pred,
                "util_pred": full_util, "logits": full_util}

    def loss(self, base_pred, delta_pred, util_pred, reward_vec, ndcg_vec, latencies):
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
        return kl + self.alpha * huber_base + self.beta * huber_delta
