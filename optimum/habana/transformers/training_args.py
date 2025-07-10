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

import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import torch.distributed as dist
from accelerate import DistributedType, PartialState
from accelerate.state import AcceleratorState
from packaging import version
from transformers.debug_utils import DebugOption
from transformers.file_utils import cached_property, is_torch_available, requires_backends
from transformers.trainer_pt_utils import AcceleratorConfig
from transformers.trainer_utils import (
    EvaluationStrategy,
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SaveStrategy,
    SchedulerType,
)
from transformers.training_args import (
    OptimizerNames,
    ParallelMode,
    TrainingArguments,
    _convert_str_dict,
    default_logdir,
)
from transformers.utils import (
    ACCELERATE_MIN_VERSION,
    get_full_repo_name,
    is_accelerate_available,
    is_safetensors_available,
    strtobool,
)

from optimum.utils import logging

from ..distributed import parallel_state
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
        compile_from_sec_iteration (`bool`, *optional*, defaults to `False`):
            Whether to torch.compile from the second training iteration.
        compile_dynamic (`bool|None`, *optional*, defaults to `None`):
            Set value of 'dynamic' parameter for torch.compile.
        use_regional_compilation (`bool`, *optional*, defaults to `False`):
            Whether to use regional compile with deepspeed
        inline_inbuilt_nn_modules (`bool`, *optional*, defaults to `None`):
            Set value of 'inline_inbuilt_nn_modules' parameter for torch._dynamo.config. Currently, disabling this parameter improves the performance of the ALBERT model.
        cache_size_limit(`int`, *optional*, defaults to 'None'):
            Set value of 'cache_size_limit' parameter for torch._dynamo.config
        allow_unspec_int_on_nn_module (`bool`, *optional*, defaults to `None`):
            Set value of 'allow_unspec_int_on_nn_module' parameter for torch._dynamo.config.
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
            Number of training steps to ignore for profiling.
        profiling_steps (`int`, *optional*, defaults to 0):
            Number of training steps to be captured when enabling profiling.
        profiling_warmup_steps_eval (`int`, *optional*, defaults to 0):
            Number of eval steps to ignore for profiling.
        profiling_steps_eval (`int`, *optional*, defaults to 0):
            Number of eval steps to be captured when enabling profiling.
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
        metadata={
            "help": (
                "Whether to use compiled autograd for training. Currently only for summarization models or when using deepspeed."
            )
        },
    )

    compile_from_sec_iteration: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to torch.compile from the second training iteration.")},
    )

    compile_dynamic: Optional[bool | None] = field(
        default=None,
        metadata={"help": ("Set value of 'dynamic' parameter for torch.compile.")},
    )

    use_zero3_leaf_promotion: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to enable leaf promotion in compile")},
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

    # This works only if compile kwarg "dynamic = None" or "dynamic = True" is
    # set and has no effect when "dynamic = False"
    allow_unspec_int_on_nn_module: Optional[bool] = field(
        default=None,
        metadata={"help": ("Set value of 'allow_unspec_int_on_nn_module' parameter for torch._dynamo.config.")},
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
        metadata={"help": ("Number of training steps to ignore for profiling.")},
    )

    profiling_steps: Optional[int] = field(
        default=0,
        metadata={"help": ("Number of training steps to be captured when enabling profiling.")},
    )

    profiling_warmup_steps_eval: Optional[int] = field(
        default=0,
        metadata={"help": ("Number of eval steps to ignore for profiling.")},
    )

    profiling_steps_eval: Optional[int] = field(
        default=0,
        metadata={"help": ("Number of eval steps to be captured when enabling profiling.")},
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
        if self.use_hpu_graphs:
            warnings.warn(
                (
                    "`--use_hpu_graphs` is deprecated and will be removed in a future version of 🤗 Optimum Habana. Use `--use_hpu_graphs_for_training` or `--use_hpu_graphs_for_inference` instead."
                ),
                FutureWarning,
            )

        use_hpu_graphs = self.use_hpu_graphs or self.use_hpu_graphs_for_inference or self.use_hpu_graphs_for_training

        if (self.use_lazy_mode or use_hpu_graphs or self.gaudi_config_name) and not self.use_habana:
            raise ValueError(
                "`--use_lazy_mode`, `--use_hpu_graphs_for_inference`, `--use_hpu_graphs_for_training` and `--gaudi_config_name` cannot be used without `--use_habana`."
            )
        if use_hpu_graphs and (not self.use_lazy_mode and not self.torch_compile_backend):
            raise ValueError(
                "`--use_hpu_graphs_for_inference` and `--use_hpu_graphs_for_training` cannot be used in eager mode. Please set `--use_lazy_mode` to True."
            )

        if self.distribution_strategy not in SUPPORTED_DISTRIBUTION_STRATEGIES:
            raise ValueError(
                f"`--distribution_strategy` is {self.distribution_strategy} which is an invalid or unsupported value. Possible choices are: {', '.join(SUPPORTED_DISTRIBUTION_STRATEGIES)}."
            )

        if self.disable_tensor_cache_hpu_graphs and not use_hpu_graphs:
            raise ValueError("must be using hpu graphs to set disable_tensor_cache_hpu_graphs.")

        if self.max_hpu_graphs is not None and not use_hpu_graphs:
            raise ValueError("must be using hpu graphs to set max_hpu_graphs.")

        # Raise errors for arguments that are not supported by optimum-habana
        if self.fp16 or self.fp16_full_eval:
            raise ValueError(
                "--fp16, --fp16_backend, --fp16_full_eval and --fp16_opt_level are not"
                " supported by optimum-habana. Mixed-precision can be enabled in your Gaudi configuration."
            )
        if self.tpu_num_cores or self.tpu_metrics_debug:
            raise ValueError("TPUs are not supported by optimum-habana.")
        if self.mp_parameters:
            raise ValueError("--mp_parameters is not supported by optimum-habana.")
        if self.tf32:
            raise ValueError("--tf32 is not supported by optimum-habana.")

        if self.throughput_warmup_steps < 0:
            raise ValueError("--throughput_warmup_steps must be positive.")

        # Set default output_dir if not provided
        if self.output_dir is None:
            self.output_dir = "trainer_output"
            logger.info(
                "No output directory specified, defaulting to 'trainer_output'. "
                "To change this behavior, specify --output_dir when creating TrainingArguments."
            )

        # Parse in args that could be `dict` sent in from the CLI as a string
        for field in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, field)
            # We only want to do this if the str starts with a bracket to indicate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, field, loaded_dict)

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        # see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.eval_strategy, EvaluationStrategy):
            warnings.warn(
                "using `EvaluationStrategy` for `eval_strategy` is deprecated and will be removed in version 5"
                " of 🤗 Transformers. Use `IntervalStrategy` instead",
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.eval_strategy = self.eval_strategy.value

        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = SaveStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.eval_strategy != IntervalStrategy.NO:
            self.do_eval = True

        if self.torch_empty_cache_steps is not None:
            if not (isinstance(self.torch_empty_cache_steps, int) and self.torch_empty_cache_steps > 0):
                raise ValueError(
                    f"`torch_empty_cache_steps` must be an integer bigger than 0, got {self.torch_empty_cache_steps}."
                )

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.eval_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.eval_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(f"--logging_steps must be an integer if bigger than 1: {self.logging_steps}")
            self.logging_steps = int(self.logging_steps)
        if self.eval_strategy == IntervalStrategy.STEPS and self.eval_steps > 1:
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(f"--eval_steps must be an integer if bigger than 1: {self.eval_steps}")
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == SaveStrategy.STEPS and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(f"--save_steps must be an integer if bigger than 1: {self.save_steps}")
            self.save_steps = int(self.save_steps)

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end and self.save_strategy != SaveStrategy.BEST:
            if self.eval_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.eval_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.eval_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                if self.eval_steps < 1 or self.save_steps < 1:
                    if not (self.eval_steps < 1 and self.save_steps < 1):
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            "steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps "
                            f"{self.save_steps} and eval_steps {self.eval_steps}."
                        )
                    # Work around floating point precision issues
                    LARGE_MULTIPLIER = 1_000_000
                    if (self.save_steps * LARGE_MULTIPLIER) % (self.eval_steps * LARGE_MULTIPLIER) != 0:
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            f"steps, but found {self.save_steps}, which is not a multiple of {self.eval_steps}."
                        )
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        safetensors_available = is_safetensors_available()
        if self.save_safetensors and not safetensors_available:
            raise ValueError(f"--save_safetensors={self.save_safetensors} requires safetensors to be installed!")
        if not self.save_safetensors and safetensors_available:
            logger.info(
                f"Found safetensors installation, but --save_safetensors={self.save_safetensors}. "
                f"Safetensors should be a preferred weights saving format due to security and performance reasons. "
                f"If your model cannot be saved by safetensors please feel free to open an issue at "
                f"https://github.com/huggingface/safetensors!"
            )

        if (
            self.load_best_model_at_end or self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU
        ) and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = not (self.metric_for_best_model.endswith("loss"))
        if self.run_name is None:
            self.run_name = self.output_dir

        if self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if self.eval_strategy == IntervalStrategy.NO:
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires an eval strategy")
            if not is_torch_available():
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires torch>=0.2.0")

        self.optim = OptimizerNames(self.optim)
        if self.adafactor:
            warnings.warn(
                (
                    "`--adafactor` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--optim"
                    " adafactor` instead"
                ),
                FutureWarning,
            )
            self.optim = OptimizerNames.ADAFACTOR
        if self.optim == OptimizerNames.ADAMW_TORCH_FUSED and is_torch_available():
            if version.parse(version.parse(torch.__version__).base_version) < version.parse("2.0.0"):
                raise ValueError("--optim adamw_torch_fused requires PyTorch 2.0 or higher")

        # We need to setup the accelerator config here *before* the first call to `self.device`
        if is_accelerate_available():
            if not isinstance(self.accelerator_config, AcceleratorConfig):
                if self.accelerator_config is None:
                    self.accelerator_config = AcceleratorConfig()
                elif isinstance(self.accelerator_config, dict):
                    self.accelerator_config = AcceleratorConfig(**self.accelerator_config)
                # Check that a user didn't pass in the class instantiator
                # such as `accelerator_config = AcceleratorConfig`
                elif isinstance(self.accelerator_config, type):
                    raise NotImplementedError(
                        "Tried passing in a callable to `accelerator_config`, but this is not supported. "
                        "Please pass in a fully constructed `AcceleratorConfig` object instead."
                    )
                else:
                    self.accelerator_config = AcceleratorConfig.from_json_file(self.accelerator_config)

            if self.dataloader_drop_last:
                self.accelerator_config.even_batches = False

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
            # Note: PT_HPU_LAZY_MODE=0 needs to be set before library is loaded,
            #       setting it here would be too late - hence assertion.
        if self.torch_compile and self.torch_compile_backend is None:
            self.torch_compile_backend = "hpu_backend"

        # accelerate integration for torch compile
        if self.torch_compile:
            # set env vars for accelerate
            prefix = "ACCELERATE_DYNAMO_"
            os.environ[prefix + "BACKEND"] = self.torch_compile_backend
            if self.torch_compile_mode is not None:
                os.environ[prefix + "MODE"] = self.torch_compile_mode
            if self.compile_dynamic is not None:
                os.environ[prefix + "USE_DYNAMIC"] = str(self.compile_dynamic)

        # if training args is specified, it will override the one specified in the accelerate config
        mixed_precision_dtype = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
        if self.fp8:
            mixed_precision_dtype = "fp8"
        elif self.bf16:
            mixed_precision_dtype = "bf16"
        os.environ["ACCELERATE_MIXED_PRECISION"] = mixed_precision_dtype

        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from transformers.integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()

            if "codecarbon" in self.report_to and torch.version.hip:
                logger.warning(
                    "When using the Trainer, CodeCarbonCallback requires the `codecarbon` package, which is not compatible with AMD ROCm (https://github.com/mlco2/codecarbon/pull/490). Automatically disabling the codecarbon callback. Reference: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments.report_to."
                )
                self.report_to.remove("codecarbon")

        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio"
                " during training"
            )

        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be of type int and must be 0 or a positive integer.")

        # Copy of https://github.com/huggingface/transformers/blob/b71f20a7c9f3716d30f6738501559acf863e2c5c/src/transformers/training_args.py#L1563
        # except following changes, (1) Remove XLA specific code & (2) change fsdp_backward_prefetch to backward_prefetch
        if isinstance(self.fsdp, bool):
            self.fsdp = [FSDPOption.FULL_SHARD] if self.fsdp else ""
        if isinstance(self.fsdp, str):
            self.fsdp = [FSDPOption(s) for s in self.fsdp.split()]
        if self.fsdp == [FSDPOption.OFFLOAD]:
            raise ValueError(
                "`--fsdp offload` can't work on its own. It needs to be added to `--fsdp full_shard` or "
                '`--fsdp shard_grad_op`. For example, `--fsdp "full_shard offload"`.'
            )
        elif FSDPOption.FULL_SHARD in self.fsdp and FSDPOption.SHARD_GRAD_OP in self.fsdp:
            raise ValueError("`--fsdp full_shard` is not compatible with `--fsdp shard_grad_op`.")

        if self.gradient_checkpointing and (
            FSDPOption.FULL_SHARD in self.fsdp or FSDPOption.HYBRID_SHARD in self.fsdp
        ):
            logger.warning(
                "When using FSDP full shard, instead of using `gradient_checkpointing` in TrainingArguments, please"
                " use `activation_checkpointing` in `fsdp_config`. The former introduces a redundant AllGather"
                " operation in backward pass. Reference: https://github.com/huggingface/transformers/issues/30404"
            )

        if self.fsdp_config is None:
            self.fsdp_config = {}

        if isinstance(self.fsdp_config, str):
            if len(self.fsdp) == 0:
                warnings.warn("`--fsdp_config` is useful only when `--fsdp` is specified.")
            with open(self.fsdp_config, encoding="utf-8") as f:
                self.fsdp_config = json.load(f)
                for k in list(self.fsdp_config.keys()):
                    if k.startswith("fsdp_"):
                        v = self.fsdp_config.pop(k)
                        self.fsdp_config[k[5:]] = v

        if self.fsdp_min_num_params > 0:
            warnings.warn("using `--fsdp_min_num_params` is deprecated. Use fsdp_config instead ", FutureWarning)

        self.fsdp_config["min_num_params"] = max(self.fsdp_config.get("min_num_params", 0), self.fsdp_min_num_params)

        # if fsdp_config["transformer_layer_cls_to_wrap"] is specified as a string, convert it to a list with a single object
        if isinstance(self.fsdp_config.get("transformer_layer_cls_to_wrap", None), str):
            self.fsdp_config["transformer_layer_cls_to_wrap"] = [self.fsdp_config["transformer_layer_cls_to_wrap"]]

        if self.fsdp_transformer_layer_cls_to_wrap is not None:
            warnings.warn(
                "using `--fsdp_transformer_layer_cls_to_wrap` is deprecated. Use fsdp_config instead ", FutureWarning
            )
            self.fsdp_config["transformer_layer_cls_to_wrap"] = self.fsdp_config.get(
                "transformer_layer_cls_to_wrap", []
            ) + [self.fsdp_transformer_layer_cls_to_wrap]

        if len(self.fsdp) == 0 and self.fsdp_config["min_num_params"] > 0:
            warnings.warn("`min_num_params` is useful only when `--fsdp` is specified.")

        if len(self.fsdp) == 0 and self.fsdp_config.get("transformer_layer_cls_to_wrap", None) is not None:
            warnings.warn("`transformer_layer_cls_to_wrap` is useful only when `--fsdp` is specified.")

        if (
            len(self.fsdp) > 0
            and self.fsdp_config["min_num_params"] > 0
            and self.fsdp_config.get("transformer_layer_cls_to_wrap", None) is not None
        ):
            raise ValueError("`min_num_params` and `transformer_layer_cls_to_wrap` are mutually exclusive.")
        self.fsdp_config["xla"] = self.fsdp_config.get("xla", False)
        self.fsdp_config["xla_fsdp_v2"] = self.fsdp_config.get("xla_fsdp_v2", False)
        self.fsdp_config["xla_fsdp_grad_ckpt"] = self.fsdp_config.get("xla_fsdp_grad_ckpt", False)

        if self.tp_size > 1:
            os.environ["ACCELERATE_USE_TP"] = "true"
            os.environ["TP_SIZE"] = str(self.tp_size)
        # accelerate integration for FSDP
        if len(self.fsdp) > 0 and not self.fsdp_config["xla"]:
            os.environ["ACCELERATE_USE_FSDP"] = "true"
            from accelerate.utils.constants import (
                FSDP_AUTO_WRAP_POLICY,
                FSDP_SHARDING_STRATEGY,
            )

            prefix = "FSDP_"
            for fsdp_option in self.fsdp:
                if fsdp_option.upper() in FSDP_SHARDING_STRATEGY:
                    # set environment variable for FSDP sharding strategy
                    os.environ[f"{prefix}SHARDING_STRATEGY"] = str(
                        FSDP_SHARDING_STRATEGY.index(fsdp_option.upper()) + 1
                    )
                elif fsdp_option == FSDPOption.OFFLOAD:
                    os.environ[f"{prefix}OFFLOAD_PARAMS"] = "true"
                elif fsdp_option == FSDPOption.AUTO_WRAP:
                    os.environ[f"{prefix}AUTO_WRAP_POLICY"] = FSDP_AUTO_WRAP_POLICY[0]
                    if self.fsdp_config["min_num_params"] > 0:
                        os.environ[f"{prefix}MIN_NUM_PARAMS"] = str(self.fsdp_config["min_num_params"])
                        os.environ[f"{prefix}AUTO_WRAP_POLICY"] = FSDP_AUTO_WRAP_POLICY[1]
                    elif self.fsdp_config.get("transformer_layer_cls_to_wrap", None) is not None:
                        os.environ[f"{prefix}TRANSFORMER_CLS_TO_WRAP"] = ",".join(
                            self.fsdp_config["transformer_layer_cls_to_wrap"]
                        )
            prefetch_policy = self.fsdp_config.get("backward_prefetch", "NO_PREFETCH")
            os.environ[f"{prefix}BACKWARD_PREFETCH"] = prefetch_policy.upper()
            os.environ[f"{prefix}FORWARD_PREFETCH"] = str(self.fsdp_config.get("forward_prefetch", "false")).lower()

            sync_module_states = str(self.fsdp_config.get("sync_module_states", "true")).lower()
            cpu_ram_efficient_loading = str(self.fsdp_config.get("cpu_ram_efficient_loading", "false")).lower()

            if sync_module_states == "false" and cpu_ram_efficient_loading == "true":
                # In this case, all the processes except the main process would have random weights leading
                # to unexpected behaviour during training, thus throwing error here to prevent it.
                raise ValueError('`sync_module_states` must be `"True"` if `cpu_ram_efficient_loading` is `"True"`')

            os.environ[f"{prefix}SYNC_MODULE_STATES"] = sync_module_states
            os.environ[f"{prefix}CPU_RAM_EFFICIENT_LOADING"] = cpu_ram_efficient_loading

            os.environ[f"{prefix}USE_ORIG_PARAMS"] = str(self.fsdp_config.get("use_orig_params", "true")).lower()

            os.environ[f"{prefix}ACTIVATION_CHECKPOINTING"] = str(
                self.fsdp_config.get("activation_checkpointing", "false")
            )

        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]
        elif self.debug is None:
            self.debug = []

        # This call to self.device is necessary to call _setup_devices so that
        # torch.distributed is initialized
        device_is_hpu = self.device.type == "hpu"
        self.deepspeed_plugin = None
        if self.deepspeed:
            if not device_is_hpu:
                raise ValueError("This version of DeepSpeed must be run on HPUs.")

            # - must be run very last in arg parsing, since it will use a lot of these settings.
            # - must be run before the model is created.
            if not is_accelerate_available():
                raise ValueError("--deepspeed requires Accelerate to be installed: `pip install accelerate`.")
            from .integrations.deepspeed import GaudiTrainerDeepSpeedConfig

            # will be used later by the Trainer
            # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
            self.hf_deepspeed_config = GaudiTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)

            # Accelerate DeepSpeed Plugin
            from accelerate.utils import DeepSpeedPlugin

            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.hf_deepspeed_config)
        elif strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")):
            # Accelerate DeepSpeed Plugin
            from accelerate.utils import DeepSpeedPlugin

            self.deepspeed_plugin = DeepSpeedPlugin()
            mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
            self.deepspeed_plugin.set_mixed_precision(mixed_precision)
            self.deepspeed_plugin.set_deepspeed_weakref()

        if self.use_cpu:
            self.dataloader_pin_memory = False

        if self.dataloader_num_workers == 0 and self.dataloader_prefetch_factor is not None:
            raise ValueError(
                "--dataloader_prefetch_factor can only be set when data is loaded in a different process, i.e."
                " when --dataloader_num_workers > 1."
            )

        if self.push_to_hub_token is not None:
            warnings.warn(
                (
                    "`--push_to_hub_token` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                    "`--hub_token` instead."
                ),
                FutureWarning,
            )
            self.hub_token = self.push_to_hub_token

        if self.push_to_hub_model_id is not None:
            self.hub_model_id = get_full_repo_name(
                self.push_to_hub_model_id, organization=self.push_to_hub_organization, token=self.hub_token
            )
            if self.push_to_hub_organization is not None:
                warnings.warn(
                    (
                        "`--push_to_hub_model_id` and `--push_to_hub_organization` are deprecated and will be removed"
                        " in version 5 of 🤗 Transformers. Use `--hub_model_id` instead and pass the full repo name to"
                        f" this argument (in this case {self.hub_model_id})."
                    ),
                    FutureWarning,
                )
            else:
                warnings.warn(
                    (
                        "`--push_to_hub_model_id` is deprecated and will be removed in version 5 of 🤗 Transformers."
                        " Use `--hub_model_id` instead and pass the full repo name to this argument (in this case"
                        f" {self.hub_model_id})."
                    ),
                    FutureWarning,
                )
        elif self.push_to_hub_organization is not None:
            self.hub_model_id = f"{self.push_to_hub_organization}/{Path(self.output_dir).name}"
            warnings.warn(
                (
                    "`--push_to_hub_organization` is deprecated and will be removed in version 5 of 🤗 Transformers."
                    " Use `--hub_model_id` instead and pass the full repo name to this argument (in this case"
                    f" {self.hub_model_id})."
                ),
                FutureWarning,
            )

        if self.eval_use_gather_object and not is_accelerate_available("0.30.0"):
            raise ValueError(
                "--eval_use_gather_object requires Accelerate to be version of `accelerate` > 0.30.0."
                "This is not supported and we recommend you to update your version."
            )

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
        # those deprecated arguments are removed from TrainingArguments. (TODO: transformers v5)
        del self_as_dict["per_gpu_train_batch_size"]
        del self_as_dict["per_gpu_eval_batch_size"]
        # Remove arguments that are unsupported by optimum-habana
        for key in UNSUPPORTED_ARGUMENTS:
            del self_as_dict[key]

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @cached_property
    def _setup_devices(self) -> "torch.device":
        requires_backends(self, ["torch"])

        # Hack to make sure bf16/fp32 ops are specified before calling habana_frameworks.torch.core
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

        if self.allow_unspec_int_on_nn_module is not None:
            torch._dynamo.config.allow_unspec_int_on_nn_module = self.allow_unspec_int_on_nn_module

        logger.info("PyTorch: setting up devices")
        if not is_accelerate_available():
            raise ImportError(
                f"Using the `Trainer` with `PyTorch` requires `accelerate>={ACCELERATE_MIN_VERSION}`: "
                f"Please run `pip install transformers[torch]` or `pip install accelerate -U`"
            )
        # We delay the init of `PartialState` to the end for clarity
        accelerator_state_kwargs = {"enabled": True, "use_configured_state": False}
        if isinstance(self.accelerator_config, AcceleratorConfig):
            accelerator_state_kwargs["use_configured_state"] = self.accelerator_config.pop(
                "use_configured_state", False
            )
        if accelerator_state_kwargs["use_configured_state"]:
            if PartialState._shared_state == {}:
                raise ValueError(
                    "Passing `'use_configured_state':True` to the AcceleratorConfig requires a pre-configured "
                    "`AcceleratorState` or `PartialState` to be defined before calling `TrainingArguments`. "
                )
            # We rely on `PartialState` to yell if there's issues here (which it will)
            self.distributed_state = PartialState(cpu=self.use_cpu)
            if self.deepspeed and self.distributed_state.distributed_type != DistributedType.DEEPSPEED:
                raise RuntimeError(
                    "Tried to use an already configured `Accelerator` or `PartialState` that was not initialized for DeepSpeed, "
                    "but also passed in a `deepspeed` configuration to the `TrainingArguments`. Please set "
                    "`use_configured_state:False` instead or setup your `Accelerator` or `PartialState` properly."
                )
        else:
            AcceleratorState._reset_state(reset_partial_state=True)
            self.distributed_state = None

        # Set the log level here for optimum.utils.logging
        # otherwise logs are not sent in this method.
        log_level = self.get_process_log_level()
        logging.set_verbosity(log_level)

        if not self.use_ipex and "ACCELERATE_USE_IPEX" not in os.environ:
            os.environ["ACCELERATE_USE_IPEX"] = "false"

        if self.minimize_memory:
            os.environ["PT_HPU_FP8_MINIMIZE_MEMORY"] = "true"

        self._n_gpu = 1
        if self.use_cpu or strtobool(os.environ.get("ACCELERATE_USE_CPU", "False")):
            accelerator_state_kwargs["cpu"] = True
            self._n_gpu = 0
        elif self.use_habana:
            # Some methods needs to be tweaked to optimally run on Gaudi
            # Calling this method here to be sure it is done before model instantiation
            # Otherwise this will fail when some __init__ methods are overridden (cf. GPT2Attention)
            from .modeling_utils import adapt_transformers_to_gaudi

            adapt_transformers_to_gaudi()

            if self.use_lazy_mode:
                logger.info("Enabled lazy mode.")
            elif not self.torch_compile:
                if os.getenv("PT_HPU_LAZY_MODE", "1") != "0":
                    raise ValueError(
                        "Lazy mode or compile mode not enabled => eager mode should be enabled using PT_HPU_LAZY_MODE=0"
                    )

            accelerator_state_kwargs["cpu"] = False
            accelerator_state_kwargs["use_deepspeed"] = self.deepspeed
            accelerator_state_kwargs["timeout"] = timedelta(seconds=self.ddp_timeout)
        else:
            raise ValueError(
                "No device has been set. Use either --use_habana to run on HPU or --use_cpu to run on CPU."
            )

        # Now we pop everything
        if accelerator_state_kwargs.pop("enabled", False) and not accelerator_state_kwargs.pop(
            "use_configured_state", False
        ):
            # We need to patch this env var when enabling to detect deepspeed
            use_deepspeed = accelerator_state_kwargs.pop("use_deepspeed", False)
            if use_deepspeed:
                os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.distributed_state = PartialState(**accelerator_state_kwargs)
            if use_deepspeed:
                del os.environ["ACCELERATE_USE_DEEPSPEED"]

        # Sequence parallelism
        if self.parallel_mode == ParallelMode.DISTRIBUTED:
            if parallel_state.is_unitialized():
                parallel_state.initialize_model_parallel(
                    sequence_parallel_size=self.context_parallel_size, use_fp8=False
                )
            else:
                if parallel_state.get_sequence_parallel_world_size() != self.context_parallel_size:
                    raise ValueError(
                        "The initialized sequence parallel world size does not match the context parallel size."
                    )
                if parallel_state.amax_reduction_is_initialized():
                    logger.info("FP8 amax reduction group is already initialized.")

        device = self.distributed_state.device
        self.local_rank = self.distributed_state.local_process_index

        if dist.is_available() and dist.is_initialized() and self.parallel_mode != ParallelMode.DISTRIBUTED:
            logger.warning(
                "torch.distributed process group is initialized, but parallel_mode != ParallelMode.DISTRIBUTED. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )

        if self.distributed_state.distributed_type == DistributedType.NO:
            self._n_gpu = 0

        return device
