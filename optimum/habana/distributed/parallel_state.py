# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

from typing import Optional

import torch


# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# FP8 amax reduction group.
_AMAX_REDUCTION_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

_TRAINING_MODE = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# For DeepSpeed's sequence parallel
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

# This group includes processes for both data and sequence parallelisms.
# We use this group to reduce gradients and shard parameters and optimizer stages for ZeRO.
_SEQUENCE_DATA_PARALLEL_GROUP = None
_SEQUENCE_DATA_PARALLEL_WORLD_SIZE = None
_SEQUENCE_DATA_PARALLEL_RANK = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    sequence_parallel_size: int = 1,
    use_fp8: bool = False,
    use_distributed_optimizer: bool = False,
) -> None:
    """Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        sequence_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a sequence parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a sequence parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and sequence parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do sequence parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Sequence parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a sequence parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of sequence parallelism on data parallel group for
            weight gradient all-reduce.

        use_fp8 (bool, default = False):
            Construct GPU groups needed for FP8 training, namely for
            amax reduction across the product of the data-parallel and
            tensor-parallel groups.

        use_distributed_optimizer (bool, default = False):
            Create a new process group using Gloo backend

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size * sequence_parallel_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x sequence_parallel_size ({sequence_parallel_size})"
        )

    enable_ds_sequence_parallel = sequence_parallel_size > 1
    if enable_ds_sequence_parallel:
        assert tensor_model_parallel_size == 1 and pipeline_model_parallel_size == 1, (
            "DeepSpeed's sequence parallel does not work with tensor parallel or pipeline parallel"
        )

        if world_size % sequence_parallel_size != 0:
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by sequence_parallel_size {sequence_parallel_size})"
            )

    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * sequence_parallel_size
    )
    sequence_data_parallel_size: int = sequence_parallel_size * data_parallel_size

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    # num_data_parallel_groups: int = world_size // data_parallel_size
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    num_sequence_data_parallel_groups: int = world_size // sequence_parallel_size // data_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError("pipeline-model-parallel size should be greater than 2 with interleaved schedule")
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"

    # Build the data-parallel groups.
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups

        if enable_ds_sequence_parallel:
            tp_or_sp_size = sequence_parallel_size
        else:
            tp_or_sp_size = tensor_model_parallel_size

        for j in range(tp_or_sp_size):
            ranks = range(start_rank + j, end_rank, tp_or_sp_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            if use_distributed_optimizer:
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            else:
                group_gloo = None
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    assert _SEQUENCE_PARALLEL_GROUP is None, "sequence parallel group is already initialized"

    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

    global _SEQUENCE_PARALLEL_WORLD_SIZE
    _SEQUENCE_PARALLEL_WORLD_SIZE = sequence_parallel_size

    global _TRAINING_MODE
    _TRAINING_MODE = True

    # Build the sequence data parallel groups.
    global _SEQUENCE_DATA_PARALLEL_GROUP
    assert _SEQUENCE_DATA_PARALLEL_GROUP is None, "sequence data parallel group is already initialized"

    all_data_sequence_parallel_group_ranks = []
    if enable_ds_sequence_parallel:
        for i in range(num_sequence_data_parallel_groups):
            ranks = range(i * sequence_data_parallel_size, (i + 1) * sequence_data_parallel_size)
            group = torch.distributed.new_group(ranks)
            all_data_sequence_parallel_group_ranks.append(list(ranks))
            if rank in ranks:
                _SEQUENCE_DATA_PARALLEL_GROUP = group
    else:
        _SEQUENCE_DATA_PARALLEL_GROUP = _DATA_PARALLEL_GROUP

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"

    num_model_parallel_groups = sequence_data_parallel_size if enable_ds_sequence_parallel else data_parallel_size
    model_parallel_group_ranks = (
        all_data_sequence_parallel_group_ranks if enable_ds_sequence_parallel else all_data_parallel_group_ranks
    )
    for i in range(num_model_parallel_groups):
        ranks = [parallel_group_ranks[i] for parallel_group_ranks in model_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, "tensor model parallel group is already initialized"

    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, "pipeline model parallel group is already initialized"
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, "embedding group is already initialized"
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, "position embedding group is already initialized"

    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank], ranks[-1]]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the FP8 groups.
    global _AMAX_REDUCTION_GROUP
    assert _AMAX_REDUCTION_GROUP is None, "FP8 amax reduction group is already initialized"

    if use_fp8:
        amax_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_amax_groups: int = world_size // amax_group_size
        for i in range(num_amax_groups):
            start_rank = i * amax_group_size
            end_rank = (i + 1) * amax_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _AMAX_REDUCTION_GROUP = group


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def is_training_mode():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    if _TRAINING_MODE is not None:
        return _TRAINING_MODE
    else:
        return False


def set_training_mode():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    global _TRAINING_MODE
    _TRAINING_MODE = True


def set_eval_mode():
    global _TRAINING_MODE
    _TRAINING_MODE = False


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or _PIPELINE_MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def sequence_parallel_is_initialized():
    """Check if sequence and data parallel groups are initialized."""
    if _SEQUENCE_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def sequence_data_parallel_is_initialized():
    """Check if sequence data parallel groups are initialized."""
    if _SEQUENCE_DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_model_parallel_world_size():
    return None


def get_model_parallel_rank():
    return 0


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GROUP is not None, "sequence parallel group is not initialized"
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_data_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_DATA_PARALLEL_GROUP is not None, "sequence data parallel group is not initialized"
    return _SEQUENCE_DATA_PARALLEL_GROUP


def set_sequence_parallel_world_size(world_size):
    """Set the sequence  parallel size"""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    _SEQUENCE_PARALLEL_WORLD_SIZE = world_size


def set_sequence_data_parallel_world_size(world_size):
    """Set the sequence  parallel size"""
    global _SEQUENCE_DATA_PARALLEL_WORLD_SIZE
    _SEQUENCE_DATA_PARALLEL_WORLD_SIZE = world_size


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    if _SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_WORLD_SIZE
    # Context Parallelism is not yet supported for eval
    if is_training_mode():
        return torch.distributed.get_world_size(group=get_sequence_parallel_group())
    else:
        return 1


def get_sequence_data_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_DATA_PARALLEL_WORLD_SIZE
    if _SEQUENCE_DATA_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_DATA_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_data_parallel_group())


def get_data_parallel_world_size():
    return get_sequence_data_parallel_world_size()


def get_data_parallel_group():
    return get_sequence_data_parallel_group()


def set_sequence_parallel_rank(rank):
    """Set sequence parallel rank."""
    global _SEQUENCE_PARALLEL_RANK
    _SEQUENCE_PARALLEL_RANK = rank


def set_sequence_data_parallel_rank(rank):
    """Set sequence parallel rank."""
    global _SEQUENCE_DATA_PARALLEL_RANK
    _SEQUENCE_DATA_PARALLEL_RANK = rank


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    # Context Parallelism is not yet supported for eval
    if is_training_mode():
        return torch.distributed.get_rank(group=get_sequence_parallel_group())
    else:
        return 0


def get_sequence_data_parallel_rank():
    """Return my rank for the sequence data parallel group."""
    global _SEQUENCE_DATA_PARALLEL_RANK
    if _SEQUENCE_DATA_PARALLEL_RANK is not None:
        return _SEQUENCE_DATA_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_data_parallel_group())


def get_sequence_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the sequence parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_sequence_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def amax_reduction_is_initialized():
    """Check if FP8 amax reduction groups are initialized."""
    if _AMAX_REDUCTION_GROUP is None:
        return False
    return True
