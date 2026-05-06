import torch
from torch import nn


class GANLoss(nn.Module):
    """BCE-with-logits adversarial loss with cached target tensor helpers."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target_is_real):
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)


def l1_loss(prediction, target):
    return nn.functional.l1_loss(prediction, target)
