# Ported from BaRISTA: https://github.com/ShanechiLab/BaRISTA
# USC research/non-profit license: LICENSES/BaRISTA-LICENSE.md

import os
import random

import numpy as np
import torch
import torch.nn as nn


def get_activation_function(activation_str):
    if activation_str.lower() == "relu":
        return nn.ReLU()
    elif activation_str.lower() == "linear":
        return lambda x: x
    elif activation_str.lower() == "gelu":
        return nn.GELU()


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set as {seed}")
