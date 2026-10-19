"""Utility Regression objective (B1).

Instead of KL onto a softmax(nDCG/T) distribution, directly regress each arm's true nDCG
from a sigmoid-activated head. Inference picks argmax of predicted nDCG.

Target: raw 5-dim nDCG vector (values in [0,1]).
Loss:   mean squared error between sigmoid(logits) and the nDCG target.
"""
import numpy as np
import torch
import torch.nn.functional as F

from train.config import NDCG_COLS
from train.objectives import Objective


def utility_loss(logits, labels):
    """MSE between sigmoid(logits) and true nDCG, averaged over arms & queries."""
    preds = torch.sigmoid(logits)
    per_arm = (preds - labels) ** 2
    return per_arm.mean()


class UtilityRegressionObjective(Objective):
    name = "utility_regression"
    label_dtype = torch.float32
    inference_activation = "sigmoid"

    def add_targets(self, df):
        # nDCG columns already exist in the data; nothing to compute.
        return df

    def targets(self, df):
        return [list(r) for r in df[NDCG_COLS].to_numpy()]

    def loss(self, logits, labels):
        return utility_loss(logits, labels)

    def eval_matrix(self, df) -> np.ndarray:
        return df[NDCG_COLS].to_numpy()
