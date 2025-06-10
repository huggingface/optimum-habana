from dataclasses import dataclass

from accelerate.utils import KwargsHandler


@dataclass
class GaudiTERecipeKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the recipe for FP8 mixed precision training with `transformer-engine`.
    Adapted from: https://github.com/huggingface/accelerate/blob/v0.27.2/src/accelerate/utils/dataclasses.py#L180
    Args:
        margin (`int`, *optional*, defaults to 0):
            The margin to use for the scaling factor computation.
        interval (`int`, *optional*, defaults to 16):
            The interval to use for how often the scaling factor is recomputed.
        fp8_format (`str`, *optional*, defaults to "HYBRID"):
            The format to use for the FP8 recipe. Must be one of `E5M2` or `HYBRID`.
        amax_history_len (`int`, *optional*, defaults to 1):
            The length of the history to use for the scaling factor computation
        amax_compute_algo (`str`, *optional*, defaults to "most_recent"):
            The algorithm to use for the scaling factor computation. Must be one of `max` or `most_recent`.
        reduce_amax (`bool`, *optional*, defaults to "False"):
            By default, if `torch.distributed` is initialized, the `amax` value for FP8
            tensors is reduced across the `fp8_group` (specified in the `fp8_autocast`
            call). This keeps the amaxes and scaling factors synced across the given
            distributed group. If set to `False`, this reduction is skipped and every
            HPU maintains local amaxes and scaling factors. To ensure results are
            numerically identical across checkpointing boundaries in this case, all
            ranks must checkpoint in order to store the local tensors.
    """

    margin: int = 0
    interval: int = 16
    fp8_format: str = "HYBRID"
    amax_compute_algo: str = "most_recent"
    amax_history_len: int = 1
    reduce_amax: bool = False

    def __post_init__(self):
        self.fp8_format = self.fp8_format.upper()
        assert self.fp8_format in ("E5M2", "HYBRID"), "Only E5M2 and HYBRID FP8 formats are currently supported."
        assert self.amax_compute_algo in (
            "max",
            "most_recent",
        ), "Only max and most_recent `amax_compute_algo` modes are currently supported."
