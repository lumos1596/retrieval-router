"""Utility + Regret objective (B2).

Same sigmoid-head utility regression as B1, plus a differentiable expected-regret term that
penalises placing probability mass on arms far below the best. Not RL: all five rewards are
known, so the expectation is exact and backprop flows through the softmax over predicted
utilities.

L = L_utility + beta * ( max_a r_a - sum_a softmax(u_a / T) * r_a )
"""
import numpy as np
import torch
import torch.nn.functional as F

from train.config import NDCG_COLS, TEMPERATURE
from train.objectives import Objective

BETA = 0.5


def utility_regret_loss(logits, labels, beta=BETA, temperature=TEMPERATURE):
    preds = torch.sigmoid(logits)
    l_utility = ((preds - labels) ** 2).mean()

    # Expected regret: best possible reward minus expected reward under the policy.
    policy = F.softmax(preds / temperature, dim=-1)
    expected_reward = (policy * labels).sum(dim=-1)
    best_reward = labels.max(dim=-1)[0]
    l_regret = (best_reward - expected_reward).mean()

    return l_utility + beta * l_regret


class UtilityRegretObjective(Objective):
    name = "utility_regret"
    label_dtype = torch.float32
    inference_activation = "sigmoid"

    def add_targets(self, df):
        return df

    def targets(self, df):
        return [list(r) for r in df[NDCG_COLS].to_numpy()]

    def loss(self, logits, labels):
        return utility_regret_loss(logits, labels)

    def eval_matrix(self, df) -> np.ndarray:
        return df[NDCG_COLS].to_numpy()
