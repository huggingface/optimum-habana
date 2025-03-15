# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from transformers.file_utils import cached_property, is_torch_available
from transformers.training_args import OptimizerNames, TrainingArguments

from optimum.utils import is_accelerate_available, logging

from ..utils import get_habana_frameworks_version
from .gaudi_configuration import GaudiConfig


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


# List of arguments that are not supported by optimum-habana
UNSUPPORTED_ARGUMENTS = [
    "fp16",
    "fp16_backend",
    "fp16_full_eval",
    "fp16_opt_level",
    "mp_parameters",
    "tf32",
    "tpu_metrics_debug",
    "tpu_num_cores",
]


# List of supported distribution strategies
SUPPORTED_DISTRIBUTION_STRATEGIES = [
    "ddp",  # default
    "fast_ddp",
]


@dataclass
class GaudiTrainingArguments(TrainingArguments):
    """
    GaudiTrainingArguments is built on top of the Tranformers' [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
    to enable deployment on Habana's Gaudi.

    Args:
        use_habana (`bool`, *optional*, defaults to `False`):
            Whether to use Habana's HPU for running the model.
        gaudi_config_name (`str`, *optional*):
            Pretrained Gaudi config name or path.
        use_lazy_mode (`bool`, *optional*, defaults to `True`):
            Whether to use lazy mode for running the model.
        use_hpu_graphs (`bool`, *optional*, defaults to `False`):
            Deprecated, use `use_hpu_graphs_for_inference` instead. Whether to use HPU graphs for performing inference.
        use_hpu_graphs_for_inference (`bool`, *optional*, defaults to `False`):
            Whether to use HPU graphs for performing inference. It will speed up latency but may not be compatible with some operations.
        use_hpu_graphs_for_training (`bool`, *optional*, defaults to `False`):
            Whether to use HPU graphs for performing inference. It will speed up training but may not be compatible with some operations.
        use_compiled_autograd (`bool`, *optional*, defaults to `False`):
            Whether to use compiled autograd for training. Currently only for summarization models.
        compile_dynamic (`bool|None`, *optional*, defaults to `None`):
            Set value of 'dynamic' parameter for torch.compile.
        use_regional_compilation (`bool`, *optional*, defaults to `False`):
            Whether to use regional compile with deepspeed
        inline_inbuilt_nn_modules (`bool`, *optional*, defaults to `None`):
            Set value of 'inline_inbuilt_nn_modules' parameter for torch._dynamo.config. Currently, disabling this parameter improves the performance of the ALBERT model.
        cache_size_limit(`int`, *optional*, defaults to 'None'):
            Set value of 'cache_size_limit' parameter for torch._dynamo.config
        disable_tensor_cache_hpu_graphs (`bool`, *optional*, defaults to `False`):
            Whether to disable tensor cache when using hpu graphs. If True, tensors won't be cached in hpu graph and memory can be saved.
        max_hpu_graphs (`int`, *optional*):
            Maximum number of hpu graphs to be cached. Reduce to save device memory.
        distribution_strategy (`str`, *optional*, defaults to `ddp`):
            Determines how data parallel distributed training is achieved. May be: `ddp` or `fast_ddp`.
        throughput_warmup_steps (`int`, *optional*, defaults to 0):
            Number of steps to ignore for throughput calculation. For example, with `throughput_warmup_steps=N`,
            the first N steps will not be considered in the calculation of the throughput. This is especially
            useful in lazy mode where the first two or three iterations typically take longer.
        adjust_throughput ('bool', *optional*, defaults to `False`):
            Whether to remove the time taken for logging, evaluating and saving from throughput calculation.
        pipelining_fwd_bwd (`bool`, *optional*, defaults to `False`):
            Whether to add an additional `mark_step` between forward and backward for pipelining
            host backward building and HPU forward computing.
        non_blocking_data_copy (`bool`, *optional*, defaults to `False`):
            Whether to enable async data copy when preparing inputs.
        profiling_warmup_steps (`int`, *optional*, defaults to 0):
            Number of steps to ignore for profiling.
        profiling_steps (`int`, *optional*, defaults to 0):
            Number of steps to be captured when enabling profiling.
    """

    use_habana: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use Habana's HPU for running the model."},
    )

    gaudi_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained Gaudi config name or path."},
    )

    use_lazy_mode: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use lazy mode for running the model."},
    )

    use_hpu_graphs: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Deprecated, use `use_hpu_graphs_for_inference` instead. Whether to use HPU graphs for performing inference."
        },
    )

    use_hpu_graphs_for_inference: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use HPU graphs for performing inference. It will speed up latency but may not be compatible with some operations."
        },
    )

    use_hpu_graphs_for_training: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use HPU graphs for performing training. It will speed up training but may not be compatible with some operations."
        },
    )

    use_compiled_autograd: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to use compiled autograd for training. Currently only for summarization models.")},
    )

    torch_compile_dynamic: Optional[bool | None] = field(
        default=None,
        metadata={"help": ("Set value of 'dynamic' parameter for torch.compile.")},
    )

    cache_size_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Set value of 'cache_size_limit' parameter for torch._dynamo.config."},
    )

    use_regional_compilation: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to use regional compile for traing.")},
    )

    inline_inbuilt_nn_modules: Optional[bool] = field(
        default=None,
        metadata={"help": ("Set value of 'inline_inbuilt_nn_modules' parameter for torch._dynamo.config.")},
    )

    disable_tensor_cache_hpu_graphs: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use a tensor cache for hpu graphs."},
    )

    max_hpu_graphs: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of HPU graphs to use."},
    )

    distribution_strategy: Optional[str] = field(
        default="ddp",
        metadata={
            "help": "Determines how distributed data parallel training is achieved. "
            "Can be either `ddp` (i.e. using `DistributedDataParallel`) or "
            "`fast_ddp` (i.e. using `optimum.habana.distributed.all_reduce_gradients`).",
            "choices": ["ddp", "fast_ddp"],
        },
    )

    context_parallel_size: Optional[int] = field(
        default=1,
        metadata={"help": ("Determines how many ranks are divided into context parallel group.")},
    )

    minimize_memory: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to enable minimze memory for fp8")},
    )

    throughput_warmup_steps: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Number of steps to ignore for throughput calculation. For example, with `throughput_warmup_steps=N`,"
                " the first N steps will not be considered in the calculation of the throughput. This is especially"
                " useful in lazy mode where the first two or three iterations typically take longer."
            )
        },
    )

    adjust_throughput: bool = field(
        default=False,
        metadata={
            "help": "Whether to remove the time taken for logging, evaluating and saving from throughput calculation."
        },
    )

    pipelining_fwd_bwd: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to add an additional `mark_step` between forward and backward for pipelining "
                "host backward building and HPU forward computing."
            )
        },
    )

    ignore_eos: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to disable stopping with eos token when calling `generate`.")},
    )

    non_blocking_data_copy: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to enable async data copy when preparing inputs.")},
    )

    profiling_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": ("Number of steps to ignore for profiling.")},
    )

    profiling_steps: Optional[int] = field(
        default=0,
        metadata={"help": ("Number of steps to be captured when enabling profiling.")},
    )

    profiling_record_shapes: Optional[bool] = field(
        default=True,
        metadata={"help": ("Record shapes when enabling profiling.")},
    )

    profiling_with_stack: Optional[bool] = field(
        default=False,
        metadata={"help": ("record source information (file and line number) for the ops when enabling profiling.")},
    )
    # Overriding the default value of optim because 'adamw_hf' is deprecated
    optim: Optional[Union[OptimizerNames, str]] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )

    # Overriding the default value of epsilon to be consistent with Habana FusedAdamW
    adam_epsilon: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Epsilon for AdamW optimizer."},
    )

    # Overriding logging_nan_inf_filter to make False the default value
    logging_nan_inf_filter: Optional[bool] = field(
        default=False,
        metadata={"help": "Filter nan and inf losses for logging."},
    )

    # Overriding ddp_bucket_cap_mb to make 230 the default value
    ddp_bucket_cap_mb: Optional[int] = field(
        default=230,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `bucket_cap_mb` passed to "
                "`DistributedDataParallel`."
            )
        },
    )

    # Overriding ddp_find_unused_parameters to make False the default value
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )

    # Overriding half_precision_backend to allow only CPU and HPU as possible mixed-precision backends for Torch Autocast.
    half_precision_backend: str = field(
        default="hpu_amp",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["cpu_amp", "hpu_amp"],
        },
    )

    # Overriding ddp_backend to replace all possible backends by hccl
    ddp_backend: Optional[str] = field(
        default="hccl",
        metadata={
            "help": "The backend to be used for distributed training.",
            "choices": ["hccl"],
        },
    )

    # Use this to override default attn_implementation in transformers
    attn_implementation: Optional[str] = field(
        default="eager",
        metadata={
            "help": "choose whether to use scale dot product attention (SDPA) or not.",
            "choices": ["eager", "sdpa"],
        },
    )

    sdp_on_bf16: bool = field(
        default=False,
        metadata={"help": "Allow pyTorch to use reduced precision in the SDPA math backend"},
    )

    fp8: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use fp8 for training."},
    )

    def __post_init__(self):
        use_hpu_graphs = self.use_hpu_graphs_for_inference or self.use_hpu_graphs_for_training

        if (use_hpu_graphs or self.use_lazy_mode or self.gaudi_config_name) and not self.use_habana:
            raise ValueError(
                "`--use_lazy_mode`, `--use_hpu_graphs_for_inference`, `--use_hpu_graphs_for_training` and `--gaudi_config_name` cannot be used without `--use_habana`."
            )
        if use_hpu_graphs and not (self.use_lazy_mode or self.torch_compile_backend):
            raise ValueError(
                "`--use_hpu_graphs_for_inference` and `--use_hpu_graphs_for_training` cannot be used in eager mode. Please set `--use_lazy_mode` to True."
            )
        if self.distribution_strategy not in SUPPORTED_DISTRIBUTION_STRATEGIES:
            raise ValueError(
                f"`--distribution_strategy` is {self.distribution_strategy} which is an invalid or unsupported value. Possible choices are: {', '.join(SUPPORTED_DISTRIBUTION_STRATEGIES)}."
            )
        if self.max_hpu_graphs is not None and not use_hpu_graphs:
            raise ValueError("must be using hpu graphs (for training or inference) to set max_hpu_graphs.")
        if self.disable_tensor_cache_hpu_graphs and not use_hpu_graphs:
            raise ValueError("must be using hpu graphs (for training or inference) to disable tensor cache.")

        if self.throughput_warmup_steps < 0:
            raise ValueError("--throughput_warmup_steps must be positive for correct throughput calculation.")

        # Disable average tokens when using single device
        if self.average_tokens_across_devices:
            try:
                if self.world_size == 1:
                    logger.warning(
                        "average_tokens_across_devices is set to True but it is invalid when world size is"
                        "1. Turn it to False automatically."
                    )
                    self.average_tokens_across_devices = False
            except ImportError as e:
                logger.warning(f"Can not specify world size due to {e}. Turn average_tokens_across_devices to False.")
                self.average_tokens_across_devices = False

        if (self.torch_compile_mode is not None or self.torch_compile_backend is not None) and not self.torch_compile:
            assert get_habana_frameworks_version().minor > 12, "Torch compile is not available"
            self.torch_compile = True
            assert not os.getenv("PT_HPU_LAZY_MODE", "1") != "0", "Dynamo and lazy are mutually exclusive."
            # Note: PT_HPU_LAZY_MODE=0 needs to be set before library is loaded, setting it here would be too late - hence assertion.

        mixed_precision_dtype = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
        if self.fp8:
            mixed_precision_dtype = "fp8"
        elif self.bf16:
            mixed_precision_dtype = "bf16"
        os.environ["ACCELERATE_MIXED_PRECISION"] = mixed_precision_dtype

        for arg in UNSUPPORTED_ARGUMENTS:
            if getattr(self, arg):
                raise ValueError(f"`--{arg}` is not supported by optimum-habana.")

        super().__post_init__()

        if self.data_seed is not None:
            if not is_accelerate_available("1.1.0"):
                raise NotImplementedError(
                    "data_seed requires Accelerate version `accelerate` >= 1.1.0. "
                    "This is not supported and we recommend you to update your version."
                )

        if self.include_inputs_for_metrics:
            logger.warning(
                "Using `include_inputs_for_metrics` is deprecated and will be removed in version 5 of 🤗 Transformers. Please use `include_for_metrics` list argument instead."
            )
            self.include_for_metrics.append("inputs")

    def __str__(self):
        self_as_dict = asdict(self)

        # Remove deprecated arguments. That code should be removed once
        # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
        del self_as_dict["per_gpu_train_batch_size"]
        del self_as_dict["per_gpu_eval_batch_size"]
        # + Remove arguments that are unsupported by optimum-habana
        for key in UNSUPPORTED_ARGUMENTS:
            del self_as_dict[key]
        ###########################################################

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    @cached_property
    def _setup_devices(self) -> "torch.device":
        # + Hack to make sure bf16/fp32 ops are specified before calling habana_frameworks.torch.core
        if self.gaudi_config_name is not None:
            gaudi_config = GaudiConfig.from_pretrained(self.gaudi_config_name)
            if (
                (self.bf16 or gaudi_config.use_torch_autocast)
                and not self.deepspeed
                and self.half_precision_backend == "hpu_amp"
            ):
                gaudi_config.declare_autocast_bf16_fp32_ops()

        if self.sdp_on_bf16:
            torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)

        if self.inline_inbuilt_nn_modules is not None:
            torch._dynamo.config.inline_inbuilt_nn_modules = self.inline_inbuilt_nn_modules

        if self.torch_compile and self.cache_size_limit is not None:
            torch._dynamo.config.cache_size_limit = self.cache_size_limit
        ###########################################################

        return super()._setup_devices
