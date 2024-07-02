# Copied from: https://github.com/InstantID/InstantID/blob/main/ip_adapter/utils.py

import torch.nn.functional as F


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")
