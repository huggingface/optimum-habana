import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .test_examples import ACCURACY_PERF_FACTOR, TIME_PERF_FACTOR
from .utils import OH_DEVICE_CONTEXT


MODELS_TO_TEST = {
    "bf16": [
        (
            "bert-base-uncased",
            "Habana/bert-base-uncased",
            "question-answering",
            24,
            8,
            "run_qa.py",
            "full_shard",
        ),
        (
            "meta-llama/Llama-2-7b-hf",
            "",
            "language-modeling",
            8,
            8,
            "run_lora_clm.py",
            "auto_wrap",
        ),
    ],
}


def _test_fsdp(
    model_name: str,
    gaudi_config: str,
    baseline,
    task: str,
    batch_size_train: int,
    batch_size_eval: int,
    script: str,
    policy: str,
    token: str,
    world_size: int = 8,
):
    os.environ["PT_HPU_LAZY_MODE"] = "0"
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"

    # Install question-answering example requirements
    cmd_line = f"pip install -r {path_to_example_dir / task / 'requirements.txt'}".split()
    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0

    command = ["python3"]

    command += [
        f"{path_to_example_dir / 'gaudi_spawn.py'}",
        "--use_mpi",
        f"--world_size {world_size}",
        f"{path_to_example_dir / task / script}",
        f"--model_name_or_path {model_name}",
        "--do_train",
        f"--per_device_eval_batch_size {batch_size_eval}",
        f"--per_device_train_batch_size {batch_size_train}",
        f"--fsdp_config {path_to_example_dir / task / 'fsdp_config.json'}",
        f"--fsdp '{policy}'",
        "--torch_compile_backend hpu_backend",
        "--torch_compile",
        "--use_habana",
    ]

    if model_name == "bert-base-uncased":
        command += [
            "--dataset_name squad",
            "--max_seq_length 384",
            "--learning_rate 3e-05",
            "--num_train_epochs 2.0",
            "--logging_steps 20",
            "--save_steps 5000",
            "--seed 42",
            "--doc_stride 128",
            "--overwrite_output_dir",
            f"--gaudi_config_name {gaudi_config}",
            "--throughput_warmup_steps 100",
            "--do_eval",
            "--sdp_on_bf16",
        ]
    else:
        command += [
            "--dataset_name tatsu-lab/alpaca ",
            "--bf16 True ",
            "--gradient_accumulation_steps 2",
            "--save_strategy 'no'",
            "--eval_strategy 'no'",
            "--learning_rate 0.0003",
            "--warmup_ratio 0.03",
            "--max_grad_norm 0.3",
            "--lr_scheduler_type 'constant'",
            "--logging_steps 1",
            "--use_lazy_mode False",
            "--pipelining_fwd_bwd False",
            "--throughput_warmup_steps 3",
            "--lora_rank 8",
            "--lora_alpha 16",
            "--lora_dropout 0.05",
            "--lora_target_modules 'q_proj' 'v_proj'",
            "--dataset_concatenation",
            "--max_seq_length 512",
            "--adam_epsilon 1e-08",
            "--low_cpu_mem_usage True",
            "--attn_softmax_bf16 True",
            "--num_train_epochs 3",
            "--use_flash_attention True",
            "--flash_attention_causal_mask True",
            f"--token {token.value}",
        ]

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        print(f"\n\nCommand to test: {' '.join(command)}\n")

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
        command = [x for y in command for x in re.split(pattern, y) if x]

        proc = subprocess.run(command)

        # Ensure the run finished without any issue
        # Use try-except to avoid logging the token if used
        try:
            assert proc.returncode == 0
        except AssertionError as e:
            if "'--token', 'hf_" in e.args[0]:
                e.args = (f"The following command failed:\n{' '.join(command[:-2])}",)
            raise

        with open(Path(tmp_dir) / "all_results.json") as fp:
            results = json.load(fp)

        # Ensure performance requirements (throughput) are met
        baseline.assertRef(
            compare=lambda actual, ref: actual >= (2 - TIME_PERF_FACTOR) * ref,
            context=[OH_DEVICE_CONTEXT],
            train_samples_per_second=results["train_samples_per_second"],
        )
        if model_name == "bert-base-uncased":
            baseline.assertRef(
                compare=lambda actual, ref: actual >= ACCURACY_PERF_FACTOR * ref,
                context=[OH_DEVICE_CONTEXT],
                eval_f1=results["eval_f1"],
            )
        else:
            baseline.assertRef(
                compare=lambda actual, ref: actual <= (2 - ACCURACY_PERF_FACTOR) * ref,
                context=[OH_DEVICE_CONTEXT],
                train_loss=results["train_loss"],
            )


@pytest.mark.skipif("gaudi1" == OH_DEVICE_CONTEXT, reason="FSDP is not supported on Gaudi1")
@pytest.mark.parametrize("model_name, gaudi_config, task, bs_train, bs_eval, script, policy", MODELS_TO_TEST["bf16"])
def test_fsdp_bf16(
    model_name: str,
    gaudi_config: str,
    task: str,
    bs_train: int,
    bs_eval: int,
    script: str,
    policy: str,
    baseline,
    token,
):
    _test_fsdp(model_name, gaudi_config, baseline, task, bs_train, bs_eval, script, policy, token)
