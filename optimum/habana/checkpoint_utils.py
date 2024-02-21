import json
import os
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers.utils import is_offline_mode


def get_repo_root(model_name_or_path, local_rank=-1, token=None):
    """
    Downloads the specified model checkpoint and returns the repository where it was downloaded.
    """
    if Path(model_name_or_path).is_dir():
        # If it is a local model, no need to download anything
        return model_name_or_path
    else:
        # Checks if online or not
        if is_offline_mode():
            if local_rank == 0:
                print("Offline mode: forcing local_files_only=True")

        # Only download PyTorch weights by default
        allow_patterns = ["*.bin"]

        # Download only on first process
        if local_rank in [-1, 0]:
            cache_dir = snapshot_download(
                model_name_or_path,
                local_files_only=is_offline_mode(),
                cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                allow_patterns=allow_patterns,
                max_workers=16,
                token=token,
            )
            if local_rank == -1:
                # If there is only one process, then the method is finished
                return cache_dir

        # Make all processes wait so that other processes can get the checkpoint directly from cache
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            allow_patterns=allow_patterns,
            token=token,
        )


def get_checkpoint_files(model_name_or_path, local_rank, token=None):
    """
    Gets the list of files for the specified model checkpoint.
    """
    cached_repo_dir = get_repo_root(model_name_or_path, local_rank=local_rank, token=token)

    # Logic for loading individual weights from https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/trainer.py#L2061
    individual_weights = [
        os.path.join(cached_repo_dir, weight_name)
        for weight_name in (
            transformers.modeling_utils.SAFE_WEIGHTS_NAME,
            transformers.modeling_utils.WEIGHTS_NAME,
        )
    ]
    checkpoint_files = []
    for weight_file in individual_weights:
        if os.path.isfile(weight_file):
            checkpoint_files.append(weight_file)
            break
    if checkpoint_files:
        return checkpoint_files

    # Code for loading sharded weights copied from https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/modeling_utils.py#L414
    index_file = os.path.join(cached_repo_dir, transformers.modeling_utils.WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(cached_repo_dir, transformers.modeling_utils.SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not safe_index_present:
        filenames = (
            transformers.modeling_utils.WEIGHTS_INDEX_NAME,
            transformers.modeling_utils.SAFE_WEIGHTS_INDEX_NAME,
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {cached_repo_dir}.")

    load_index = safe_index_file if safe_index_present else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    file_list = set(index["weight_map"].values())
    return [os.path.join(cached_repo_dir, entry) for entry in file_list]


def write_checkpoints_json(model_name_or_path, local_rank, f, token=None):
    """
    Dumps metadata into a JSON file for DeepSpeed-inference.
    """
    checkpoint_files = get_checkpoint_files(model_name_or_path, local_rank, token)
    data = {"type": "ds_model", "checkpoints": checkpoint_files, "version": 1.0}
    json.dump(data, f)
    f.flush()


def model_on_meta(config):
    """
    Checks if load the model to meta.
    """
    return config.model_type in ["bloom", "llama", "falcon"]


def get_optimized_model_name(config):
    from .transformers.generation import MODELS_OPTIMIZED_WITH_STATIC_SHAPES

    for model_type in MODELS_OPTIMIZED_WITH_STATIC_SHAPES:
        if model_type == config.model_type:
            return model_type

    return None


def model_is_optimized(config):
    """
    Checks if the given config belongs to a model in optimum/habana/transformers/models, which has a
    new input token_idx.
    """
    return get_optimized_model_name(config) is not None


def get_ds_injection_policy(config):
    model_type = get_optimized_model_name(config)
    policy = {}
    if model_type:
        if model_type == "bloom":
            from transformers.models.bloom.modeling_bloom import BloomBlock

            policy = {BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}

        if model_type == "opt":
            from transformers.models.opt.modeling_opt import OPTDecoderLayer

            policy = {OPTDecoderLayer: ("self_attn.out_proj", ".fc2")}

        if model_type == "gpt2":
            from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

            policy = {GPT2MLP: ("attn.c_proj", "mlp.c_proj")}

        if model_type == "gptj":
            from transformers.models.gptj.modeling_gptj import GPTJBlock

            policy = {GPTJBlock: ("attn.out_proj", "mlp.fc_out")}

        if model_type == "gpt_neox":
            from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

            policy = {GPTNeoXLayer: ("attention.dense", "mlp.dense_4h_to_h")}

        if model_type == "llama":
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer

            policy = {LlamaDecoderLayer: ("self_attn.o_proj", "mlp.down_proj")}

    return policy
