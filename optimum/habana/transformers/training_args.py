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
import warnings
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

from packaging import version
from transformers.debug_utils import DebugOption
from transformers.file_utils import cached_property, is_torch_available, requires_backends
from transformers.trainer_utils import EvaluationStrategy, HubStrategy, IntervalStrategy, SchedulerType
from transformers.training_args import (
    OptimizerNames,
    ParallelMode,
    TrainingArguments,
    default_logdir,
)
from transformers.utils import (
    get_full_repo_name,
    is_accelerate_available,
    is_safetensors_available,
    strtobool,
)

from optimum.utils import logging

from ..accelerate.state import GaudiAcceleratorState, GaudiPartialState
from ..accelerate.utils import GaudiDistributedType
from .gaudi_configuration import GaudiConfig


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


# List of arguments that are not supported by optimum-habana
UNSUPPORTED_ARGUMENTS = [
    "bf16_full_eval",
    "fp16",
    "fp16_backend",
    "fp16_full_eval",
    "fp16_opt_level",
    "fsdp",
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
        use_lazy_mode (`bool`, *optional*, defaults to `False`):
            Whether to use lazy mode for running the model.
        use_hpu_graphs (`bool`, *optional*, defaults to `False`):
            Deprecated, use `use_hpu_graphs_for_inference` instead. Whether to use HPU graphs for performing inference.
        use_hpu_graphs_for_inference (`bool`, *optional*, defaults to `False`):
            Whether to use HPU graphs for performing inference. It will speed up latency but may not be compatible with some operations.
        use_hpu_graphs_for_training (`bool`, *optional*, defaults to `False`):
            Whether to use HPU graphs for performing inference. It will speed up training but may not be compatible with some operations.
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
            Number of steps to ignore for profling.
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
        default=False,
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

    distribution_strategy: Optional[str] = field(
        default="ddp",
        metadata={
            "help": "Determines how distributed data parallel training is achieved. "
            "Can be either `ddp` (i.e. using `DistributedDataParallel`) or "
            "`fast_ddp` (i.e. using `optimum.habana.distributed.all_reduce_gradients`).",
            "choices": ["ddp", "fast_ddp"],
        },
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

    non_blocking_data_copy: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to enable async data copy when preparing inputs.")},
    )

    profiling_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": ("Number of steps to ignore for profling.")},
    )

    profiling_steps: Optional[int] = field(
        default=0,
        metadata={"help": ("Number of steps to be captured when enabling profiling.")},
    )

    profiling_record_shapes: Optional[bool] = field(
        default=True,
        metadata={"help": ("Record shapes when enabling profiling.")},
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

    def __post_init__(self):
        if self.use_hpu_graphs:
            warnings.warn(
                (
                    "`--use_hpu_graphs` is deprecated and will be removed in a future version of 🤗 Optimum Habana. Use "
                    "`--use_hpu_graphs_for_inference` instead."
                ),
                FutureWarning,
            )

        use_hpu_graphs = self.use_hpu_graphs or self.use_hpu_graphs_for_inference or self.use_hpu_graphs_for_training

        if (self.use_lazy_mode or use_hpu_graphs or self.gaudi_config_name) and not self.use_habana:
            raise ValueError(
                "`--use_lazy_mode`, `--use_hpu_graphs_for_inference`, `--use_hpu_graphs_for_training` and `--gaudi_config_name` cannot be used without `--use_habana`."
            )

        if use_hpu_graphs and not self.use_lazy_mode:
            raise ValueError(
                "`--use_hpu_graphs_for_inference` and `--use_hpu_graphs_for_training` cannot be used in eager mode. Please set `--use_lazy_mode` to True."
            )

        if self.distribution_strategy not in SUPPORTED_DISTRIBUTION_STRATEGIES:
            raise ValueError(
                f"`--distribution_strategy` is {self.distribution_strategy} which is an invalid or unsupported value. Possible choices are: {', '.join(SUPPORTED_DISTRIBUTION_STRATEGIES)}."
            )

        # Raise errors for arguments that are not supported by optimum-habana
        if self.bf16_full_eval:
            raise ValueError("--bf16_full_eval is not supported by optimum-habana.")
        if self.fp16 or self.fp16_full_eval:
            raise ValueError(
                "--fp16, --fp16_backend, --fp16_full_eval and --fp16_opt_level are not"
                " supported by optimum-habana. Mixed-precision can be enabled in your Gaudi configuration."
            )
        if self.fsdp:
            raise ValueError("--fsdp is not supported by optimum-habana.")
        if self.tpu_num_cores or self.tpu_metrics_debug:
            raise ValueError("TPUs are not supported by optimum-habana.")
        if self.mp_parameters:
            raise ValueError("--mp_parameters is not supported by optimum-habana.")
        if self.tf32:
            raise ValueError("--tf32 is not supported by optimum-habana.")

        if self.throughput_warmup_steps < 0:
            raise ValueError("--throughput_warmup_steps must be positive.")

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

        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn(
                (
                    "using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version"
                    " 5 of 🤗 Transformers. Use `IntervalStrategy` instead"
                ),
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.evaluation_strategy = self.evaluation_strategy.value

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(f"--logging_steps must be an integer if bigger than 1: {self.logging_steps}")
            self.logging_steps = int(self.logging_steps)
        if self.evaluation_strategy == IntervalStrategy.STEPS and self.eval_steps > 1:
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(f"--eval_steps must be an integer if bigger than 1: {self.eval_steps}")
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == IntervalStrategy.STEPS and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(f"--save_steps must be an integer if bigger than 1: {self.save_steps}")
            self.save_steps = int(self.save_steps)

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                if self.eval_steps < 1 or self.save_steps < 1:
                    if not (self.eval_steps < 1 and self.save_steps < 1):
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            "steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps"
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
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir

        if self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if self.evaluation_strategy == IntervalStrategy.NO:
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

        # if training args is specified, it will override the one specified in the accelerate config
        mixed_precision_dtype = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
        if self.bf16:
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

        logger.info("PyTorch: setting up devices")
        if not is_accelerate_available(min_version="0.21.0"):
            raise ImportError(
                "Using the `GaudiTrainer` requires `accelerate>=0.21.0`: Please run `pip install accelerate -U`."
            )
        GaudiAcceleratorState._reset_state()
        GaudiPartialState._reset_state()
        self.distributed_state = None

        # Set the log level here for optimum.utils.logging
        # otherwise logs are not sent in this method.
        log_level = self.get_process_log_level()
        logging.set_verbosity(log_level)

        if not self.use_ipex and "ACCELERATE_USE_IPEX" not in os.environ:
            os.environ["ACCELERATE_USE_IPEX"] = "false"
        if self.use_cpu or strtobool(os.environ.get("ACCELERATE_USE_CPU", "False")):
            self.distributed_state = GaudiPartialState(cpu=True, backend=self.ddp_backend)
            self._n_gpu = 0
        elif self.use_habana:
            # Some methods needs to be tweaked to optimally run on Gaudi
            # Calling this method here to be sure it is done before model instantiation
            # Otherwise this will fail when some __init__ methods are overridden (cf. GPT2Attention)
            from .modeling_utils import adapt_transformers_to_gaudi

            adapt_transformers_to_gaudi()

            if self.use_lazy_mode:
                logger.info("Enabled lazy mode.")
            else:
                os.environ["PT_HPU_LAZY_MODE"] = "2"
                logger.info("Enabled eager mode because use_lazy_mode=False.")

            if self.deepspeed:
                # Need to do similar for Accelerator init
                os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
                self.distributed_state = GaudiPartialState(timeout=timedelta(seconds=self.ddp_timeout))
                del os.environ["ACCELERATE_USE_DEEPSPEED"]
            else:
                self.distributed_state = GaudiPartialState(
                    backend=self.ddp_backend, timeout=timedelta(seconds=self.ddp_timeout)
                )
            self._n_gpu = 1
        else:
            raise ValueError(
                "No device has been set. Use either --use_habana to run on HPU or --no_cuda to run on CPU."
            )

        device = self.distributed_state.device
        self.local_rank = self.distributed_state.local_process_index
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.parallel_mode != ParallelMode.DISTRIBUTED
        ):
            logger.warning(
                "torch.distributed process group is initialized, but parallel_mode != ParallelMode.DISTRIBUTED. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )

        if self.distributed_state.distributed_type == GaudiDistributedType.NO:
            self._n_gpu = 0

        return device
