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
    TrainingArguments,
    default_logdir,
    get_int_from_env,
)
from transformers.utils import (
    ccl_version,
    get_full_repo_name,
    is_accelerate_available,
    is_psutil_available,
    is_safetensors_available,
)

from optimum.utils import logging


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
    "sharded_ddp",
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
            "help": "The backend to use for half precision.",
            "choices": ["cpu_amp", "hpu_amp"],
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
        if self.sharded_ddp:
            raise ValueError("--sharded_ddp is not supported by optimum-habana.")
        if self.tf32:
            raise ValueError("--tf32 is not supported by optimum-habana.")

        if self.throughput_warmup_steps < 0:
            raise ValueError("--throughput_warmup_steps must be positive.")

        # Handle --use_env option in torch.distributed.launch (local_rank not passed as an arg then).
        # This needs to happen before any call to self.device or self.n_gpu.
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank:
            self.local_rank = env_local_rank

        if self.local_rank != -1 and self.use_hpu_graphs_for_training and self.distribution_strategy != "fast_ddp":
            raise ValueError(
                "`--use_hpu_graphs_for_training` may only be used with `--distribution_strategy fast_ddp`"
            )

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

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
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

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir

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

        # This call to self.device is necessary to call _setup_devices so that
        # torch.distributed is initialized
        device_is_hpu = self.device.type == "hpu"
        if self.deepspeed:
            if not device_is_hpu:
                raise ValueError("This version of DeepSpeed must be run on HPUs.")

            # - must be run very last in arg parsing, since it will use a lot of these settings.
            # - must be run before the model is created.
            if not is_accelerate_available():
                raise ValueError("--deepspeed requires Accelerate to be installed: `pip install accelerate`.")
            from .deepspeed import GaudiTrainerDeepSpeedConfig

            # will be used later by the Trainer
            # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
            self.hf_deepspeed_config = GaudiTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)

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

        # Set the log level here for optimum.utils.logging
        # otherwise logs are not sent in this method.
        log_level = self.get_process_log_level()
        logging.set_verbosity(log_level)

        logger.info("PyTorch: setting up devices")
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
            logger.warning("torch.distributed process group is initialized, but local_rank == -1. ")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
            self.local_rank = get_int_from_env(
                ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"],
                self.local_rank,
            )
            if self.local_rank != -1 and not torch.distributed.is_initialized():
                # Initializes distributed backend for cpu
                if self.xpu_backend not in ("mpi", "ccl", "gloo"):
                    raise ValueError(
                        "CPU distributed training backend is not properly set. "
                        "Please set '--xpu_backend' to either 'mpi' or 'ccl' or 'gloo'."
                    )
                if self.xpu_backend == "ccl":
                    requires_backends(self, "oneccl_bind_pt")
                    if ccl_version >= "1.12":
                        import oneccl_bindings_for_pytorch  # noqa: F401
                    else:
                        import torch_ccl  # noqa: F401
                    if int(os.environ.get("CCL_WORKER_COUNT", 0)) < 1:
                        raise ValueError(
                            "CPU distributed training backend is ccl. but CCL_WORKER_COUNT is not correctly set. "
                            "Please use like 'export CCL_WORKER_COUNT = 1' to set."
                        )

                # Try to get launch configuration from environment variables set by MPI launcher - works for Intel MPI, OpenMPI and MVAPICH
                rank = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
                size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1)
                local_size = get_int_from_env(
                    ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
                )
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(size)
                os.environ["LOCAL_RANK"] = str(self.local_rank)
                if not os.environ.get("MASTER_PORT", None):
                    os.environ["MASTER_PORT"] = "29500"
                if not os.environ.get("MASTER_ADDR", None):
                    if local_size != size or self.xpu_backend != "mpi":
                        raise ValueError(
                            "Looks like distributed multinode run but MASTER_ADDR env not set, "
                            "please try exporting rank 0's hostname as MASTER_ADDR"
                        )
                if (
                    torch.get_num_threads() == 1
                    and get_int_from_env(["OMP_NUM_THREADS", "MKL_NUM_THREADS"], 0) == 0
                    and is_psutil_available()
                ):
                    import psutil

                    num_cpu_threads_per_process = int(psutil.cpu_count(logical=False) / local_size)
                    if num_cpu_threads_per_process == 0:
                        num_cpu_threads_per_process = 1
                    torch.set_num_threads(num_cpu_threads_per_process)
                    logger.info(
                        f"num_cpu_threads_per_process unset, we set it at {num_cpu_threads_per_process} to improve oob"
                        " performance."
                    )
                torch.distributed.init_process_group(
                    backend=self.xpu_backend, rank=rank, world_size=size, timeout=self.ddp_timeout_delta
                )
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

            device = torch.device("hpu")
            self._n_gpu = 1

            from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

            world_size, rank, self.local_rank = initialize_distributed_hpu()

            if self.deepspeed:
                # deepspeed inits torch.distributed internally
                from transformers.deepspeed import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError(
                        "--deepspeed requires deepspeed: `pip install"
                        " git+https://github.com/HabanaAI/DeepSpeed.git@1.10.0`."
                    )
                import deepspeed

                if world_size > 1:
                    os.environ["HLS_MODULE_ID"] = str(self.local_rank)
                    os.environ["ID"] = str(rank)

                deepspeed.init_distributed(dist_backend="hccl", timeout=timedelta(seconds=self.ddp_timeout))
                logger.info("DeepSpeed is enabled.")
            else:
                if self.local_rank != -1:
                    if not torch.distributed.is_initialized():
                        torch.distributed.init_process_group(backend="hccl", rank=rank, world_size=world_size)
                        logger.info("Enabled distributed run.")
                else:
                    logger.info("Single-device run.")
        else:
            raise ValueError(
                "No device has been set. Use either --use_habana to run on HPU or --no_cuda to run on CPU."
            )

        return device
