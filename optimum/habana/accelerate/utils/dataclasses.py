from enum import Enum


class GaudiDistributedType(str, Enum):
    """
    Represents a type of distributed environment.
    Adapted from: https://github.com/huggingface/accelerate/blob/8514c35192ac9762920f1ab052e5cea4c0e46eeb/src/accelerate/utils/dataclasses.py#L176

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_HPU** -- Distributed on multiple HPUs.
        - **DEEPSPEED** -- Using DeepSpeed.
    """

    # Subclassing str as well as Enum allows the `GaudiDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_HPU = "MULTI_HPU"
    DEEPSPEED = "DEEPSPEED"
