import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    def __init__(
        self,
        probs=None,
        logits=None,
        validate_args=None,
        masks=[],
        mask_value=None,
    ):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
