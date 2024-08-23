from types import MethodType

import torch
from accelerate.utils.constants import FSDP_PYTORCH_VERSION
from accelerate.utils.imports import is_deepspeed_available, is_torch_distributed_available
from accelerate.utils.other import is_compiled_module
from accelerate.utils.transformer_engine import convert_model
from accelerate.utils.versions import is_torch_version


def extract_model_from_parallel(model, keep_fp32_wrapper: bool = True, recursive: bool = False):
    """
    Adapted from: https://github.com/huggingface/accelerate/blob/v0.33.0/src/accelerate/utils/other.py#L56

    Changes:
    - add a `distributed_model` variable to keep track of the distributed wrapper
      and not lose it when setting it back at the end (for compiled models)

    See https://github.com/huggingface/optimum-habana/pull/1281 for more information.
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    is_compiled = is_compiled_module(model)
    if is_compiled:
        compiled_model = model
        model = model._orig_mod

    if is_deepspeed_available():
        from deepspeed import DeepSpeedEngine

        options += (DeepSpeedEngine,)

    if is_torch_version(">=", FSDP_PYTORCH_VERSION) and is_torch_distributed_available():
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

        options += (FSDP,)

    # Keep track of the distributed wrapper
    # TODO: to revisit as lines 44 to 71 are now useless
    distributed_model = model
    while isinstance(model, options):
        model = model.module

    if recursive:
        # This is needed in cases such as using FSDPv2 on XLA
        def _recursive_unwrap(module):
            # Wrapped modules are standardly wrapped as `module`, similar to the cases earlier
            # with DDP, DataParallel, DeepSpeed, and FSDP
            if hasattr(module, "module"):
                unwrapped_module = _recursive_unwrap(module.module)
            else:
                unwrapped_module = module
            # Next unwrap child sublayers recursively
            for name, child in unwrapped_module.named_children():
                setattr(unwrapped_module, name, _recursive_unwrap(child))
            return unwrapped_module

        # Start with top-level
        model = _recursive_unwrap(model)

    if not keep_fp32_wrapper:
        forward = model.forward
        original_forward = model.__dict__.pop("_original_forward", None)
        if original_forward is not None:
            while hasattr(forward, "__wrapped__"):
                forward = forward.__wrapped__
                if forward == original_forward:
                    break
            model.forward = MethodType(forward, model)
        if getattr(model, "_converted_to_transformer_engine", False):
            convert_model(model, to_transformer_engine=False)

    if is_compiled:
        compiled_model._orig_mod = distributed_model
        model = compiled_model

    return model
