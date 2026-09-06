# Ported from BaRISTA: https://github.com/ShanechiLab/BaRISTA
# USC research/non-profit license: LICENSES/BaRISTA-LICENSE.md

import torch.nn as nn


def get_activation_function(activation_str):
    if activation_str.lower() == "relu":
        return nn.ReLU()
    elif activation_str.lower() == "linear":
        return lambda x: x
    elif activation_str.lower() == "gelu":
        return nn.GELU()
