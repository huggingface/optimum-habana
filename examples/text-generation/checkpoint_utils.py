import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers.utils import is_offline_mode


def get_repo_root(model_name_or_path, local_rank):
    """
    Downloads the specified model checkpoint and returns the repository where it was downloaded.
    """
    # Checks if online or not
    if is_offline_mode():
        if local_rank == 0:
            print("Offline mode: forcing local_files_only=True")

    # Download only on first process
    if local_rank == 0:
        snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            ignore_patterns=["*.safetensors"],
        )

    # Make all processes wait so that other processes can get the checkpoint directly from cache
    torch.distributed.barrier()

    return snapshot_download(
        model_name_or_path,
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
        ignore_patterns=["*.safetensors"],
    )


def get_checkpoint_files(model_name_or_path, local_rank):
    """
    Gets the list of files for the specified model checkpoint.
    """
    cached_repo_dir = get_repo_root(model_name_or_path, local_rank)

    # Extensions: .bin | .pt
    # Creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
    return file_list


def write_checkpoints_json(model_name_or_path, local_rank, checkpoints_json):
    """
    Dumps metadata into a JSON file for DeepSpeed-inference.
    """
    checkpoint_files = get_checkpoint_files(model_name_or_path, local_rank)
    if local_rank == 0:
        data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}
        with open(checkpoints_json, "w") as fp:
            json.dump(data, fp)


def model_is_bloom(config):
    """
    Checks if the given config belongs to a BLOOM-like model.
    """
    return "bloom" in config.architectures[0].lower()


def model_is_optimized(config):
    """
    Checks if the given config belongs to a model in optimum/habana/transformers/models, which has a
    new input token_idx.
    """
    optimized_models = ["bloom", "gpt2", "opt", "gptj"]
    for optimized_model in optimized_models:
        if optimized_model in config.architectures[0].lower():
            return True
    return False
