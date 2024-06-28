import os
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from optimum.habana.transformers.models.llama.modeling_llama import (
    GaudiLlamaMLP, 
    TPGaudiLlamaMLP,
    GaudiLlamaAttention,
    TPGaudiLlamaAttention
)

# this probably belongs somewhere else but can't go in fms.distribtued b/c
# circular dependency.
def _tp_wrapped(module: nn.Module, layer: int, group: ProcessGroup):
    if hasattr(module, "to_tp"):
        return module.to_tp(group)
    elif isinstance(module, GaudiLlamaAttention):
        return TPGaudiLlamaAttention.import_module(module,layer, group)
    elif isinstance(module, GaudiLlamaMLP):
        return TPGaudiLlamaMLP.import_module(module, group)
    else:
        return module


def apply_tp(model: nn.Module, layer_idx: int, group: ProcessGroup):
    wrapped = _tp_wrapped(model, layer_idx,  group)
    if wrapped is not model:
        return wrapped

    for name, layer in model.named_children():
        tp_layer = apply_tp(layer, layer_idx, group)
        setattr(model, name, tp_layer)
    return model
