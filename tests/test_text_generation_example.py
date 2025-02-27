import json
import operator
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pytest

from optimum.habana.utils import set_seed

from .test_examples import TIME_PERF_FACTOR
from .utils import OH_DEVICE_CONTEXT


prev_quant_model_name = None
prev_quant_rank = 0

if OH_DEVICE_CONTEXT in ["gaudi2"]:
    MODELS_TO_TEST = {
        "bf16_1x": [
            ("bigscience/bloomz-7b1", 1, False, False),
            ("gpt2-xl", 1, False, False),
            ("EleutherAI/gpt-j-6b", 1, False, False),
            ("EleutherAI/gpt-neox-20b", 1, False, False),
            ("meta-llama/Llama-2-7b-hf", 1, True, True),
            ("tiiuae/falcon-40b", 1, True, False),
            ("bigcode/starcoder", 256, True, True),
            ("Salesforce/codegen2-1B", 1, False, False),
            ("mosaicml/mpt-30b", 1, False, False),
            ("mistralai/Mistral-7B-v0.1", 1, True, True),
            ("mistralai/Mixtral-8x7B-v0.1", 1, False, True),
            ("microsoft/phi-2", 1, False, False),
            ("meta-llama/Meta-Llama-3-8B", 1, True, False),
            ("meta-llama/Llama-2-7b-hf", 512, True, False),
            ("meta-llama/Llama-2-7b-hf", 512, False, False),  # in some cases like TGI, reuse_cache isn't used
            ("stabilityai/stablelm-2-12b", 1, False, False),
            ("codellama/CodeLlama-34b-hf", 1, True, False),
            ("bigcode/starcoder2-3b", 1, False, True),
            ("adept/persimmon-8b-base", 4, False, False),
            # ("Qwen/Qwen1.5-7B", 4, False, False),
            ("google/gemma-7b", 1, False, True),
            ("google/gemma-2-9b", 1, False, True),
            ("google/gemma-2-27b", 1, False, True),
            ("state-spaces/mamba-130m-hf", 1536, False, False),
            # ("Deci/DeciLM-7B", 1, False, False),
            ("Qwen/Qwen2-7B", 256, False, True),
            ("Qwen/Qwen1.5-MoE-A2.7B", 1, True, False),
            # ("EleutherAI/gpt-neo-2.7B", 1, False, False),
            # ("facebook/xglm-1.7B", 1, False, False),
            # ("CohereForAI/c4ai-command-r-v01", 1, False, False),
            ("tiiuae/falcon-mamba-7b", 1, False, False),
            ("openbmb/MiniCPM3-4B", 1, False, False),
            ("baichuan-inc/Baichuan2-7B-Chat", 1, True, False),
            ("baichuan-inc/Baichuan2-13B-Chat", 1, False, False),
            ("deepseek-ai/DeepSeek-V2-Lite", 1, False, False),
            ("THUDM/chatglm2-6b", 1, True, False),
            ("THUDM/chatglm3-6b", 1, True, False),
            ("Qwen/Qwen2.5-7B", 4, False, False),
        ],
        "fp8": [
            ("tiiuae/falcon-180B", 4, 950, True, 128, 128),
            ("meta-llama/Llama-2-7b-hf", 1, 1230, False, 128, 128),
            ("meta-llama/Llama-2-7b-hf", 1, 163, False, 128, 2048),
            ("meta-llama/Llama-2-7b-hf", 1, 94, False, 2048, 128),
            ("meta-llama/Llama-2-7b-hf", 1, 81, False, 2048, 2048),
            ("meta-llama/Llama-2-70b-hf", 4, 3042, False, 128, 128),
            ("meta-llama/Llama-2-70b-hf", 4, 750, False, 128, 2048),
            ("meta-llama/Llama-2-70b-hf", 4, 207, False, 2048, 128),
            ("meta-llama/Llama-2-70b-hf", 8, 172, False, 2048, 2048),
            ("mistralai/Mistral-7B-Instruct-v0.2", 1, 896, True, 128, 128),
            # ("mistralai/Mistral-7B-Instruct-v0.2", 1, 120, True, 128, 2048),
            # ("mistralai/Mistral-7B-Instruct-v0.2", 1, 120, True, 2048, 128),
            ("mistralai/Mistral-7B-Instruct-v0.2", 1, 44, True, 2048, 2048),
            ("mistralai/Mixtral-8x7B-v0.1", 1, 1, True, 128, 128),
            ("mistralai/Mixtral-8x7B-v0.1", 2, 768, True, 128, 128),
            # ("mistralai/Mixtral-8x7B-v0.1", 2, 96, True, 128, 2048),
            # ("mistralai/Mixtral-8x7B-v0.1", 2, 96, True, 2048, 128),
            ("mistralai/Mixtral-8x7B-v0.1", 2, 48, True, 2048, 2048),
            ("microsoft/phi-2", 1, 1, True, 128, 128),
        ],
        "load_quantized_model_with_autogptq": [
            ("TheBloke/Llama-2-7b-Chat-GPTQ", 1, 10, False, 128, 2048),
        ],
        "load_quantized_model_with_autoawq": [
            ("TheBloke/Llama-2-7b-Chat-AWQ", 1, 10, False, 128, 2048),
        ],
        "deepspeed": [
            ("bigscience/bloomz", 8, 1),
            # ("meta-llama/Llama-2-70b-hf", 8, 1),
            ("meta-llama/Meta-Llama-3-70B-Instruct", 8, 1),
            ("facebook/opt-66b", 2, 1),
            ("google/gemma-2-9b", 8, 1),
            ("Qwen/Qwen2.5-72B", 2, 1),
            ("google/gemma-2-27b", 8, 1),
        ],
        "torch_compile": [
            "meta-llama/Llama-2-7b-hf",
        ],
        "torch_compile_distributed": [
            "meta-llama/Llama-2-7b-hf",
        ],
        "distributed_tp": [
            "meta-llama/Llama-2-7b-hf",
        ],
        "contrastive_search": [
            ("gpt2-xl", 1, False),
        ],
        "beam_search": [
            ("Qwen/Qwen2-7b-Instruct", 1, True),
        ],
    }
else:
    # Gaudi1 CI
    MODELS_TO_TEST = {
        "bf16_1x": [
            ("bigscience/bloomz-7b1", 1, False, False),
            ("gpt2-xl", 1, False, False),
            # TODO: fix OPT 6.7B
            # ("facebook/opt-6.7b", 0.0),
            ("EleutherAI/gpt-j-6b", 1, True, False),
            ("meta-llama/Llama-2-7b-hf", 1, True, False),
            ("tiiuae/falcon-7b", 1, True, False),
            ("bigcode/starcoder", 1, False, False),
            ("Salesforce/codegen2-1B", 1, False, False),
            ("mosaicml/mpt-7b", 1, False, False),
            ("mistralai/Mistral-7B-v0.1", 1, True, False),
            ("microsoft/phi-2", 1, False, False),
            ("google/gemma-7b", 1, False, False),
            ("stabilityai/stablelm-2-12b", 1, False, False),
            ("Qwen/Qwen1.5-7B", 1, False, False),
            ("adept/persimmon-8b-base", 1, False, False),
            ("bigcode/starcoder2-3b", 1, False, False),
            ("state-spaces/mamba-130m-hf", 224, False, False),
        ],
        "fp8": [],
        "load_quantized_model_with_autogptq": [],
        "load_quantized_model_with_autoawq": [],
        "deepspeed": [
            ("bigscience/bloomz-7b1", 8, 1),
        ],
        "torch_compile": [],
        "torch_compile_distributed": [],
        "distributed_tp": [],
        "contrastive_search": [
            ("gpt2-xl", 1, False),
        ],
        "beam_search": [],
    }


def _test_text_generation(
    model_name: str,
    baseline,
    token: str,
    batch_size: int = 1,
    reuse_cache: bool = False,
    deepspeed: bool = False,
    world_size: int = 8,
    torch_compile: bool = False,
    fp8: bool = False,
    load_quantized_model_with_autogptq: bool = False,
    load_quantized_model_with_autoawq: bool = False,
    max_input_tokens: int = 0,
    max_output_tokens: int = 100,
    parallel_strategy: str = None,
    contrastive_search: bool = False,
    num_beams: int = 1,
    num_return_sequences: int = 1,
    check_output: bool = False,
):
    command = ["python3"]
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    env_variables = os.environ.copy()

    if deepspeed:
        command += [
            f"{path_to_example_dir / 'gaudi_spawn.py'}",
            "--use_deepspeed",
            f"--world_size {world_size}",
        ]
    elif parallel_strategy == "tp":
        command += [
            f"{path_to_example_dir / 'gaudi_spawn.py'}",
            f"--world_size {world_size}",
        ]

    command += [
        f"{path_to_example_dir / 'text-generation' / 'run_generation.py'}",
        f"--model_name_or_path {model_name}",
        f"--batch_size {batch_size}",
        "--use_kv_cache",
        f"--max_new_tokens {max_output_tokens}",
    ]

    is_starcoder_first_gen_model = "starcoder" in model_name.lower() and "starcoder2" not in model_name.lower()

    if "llama" in model_name.lower():
        command += ["--trim_logits", "--attn_softmax_bf16"]

    if "falcon" in model_name.lower() or "starcoder2" in model_name.lower():
        command += ["--use_flash_attention", "--flash_attention_causal_mask"]

    if is_starcoder_first_gen_model:
        command += ["--use_flash_attention"]

        # starcoder doesn't support reuse_cache, but implements bucket_internal instead
        if reuse_cache:
            command += ["--bucket_size 128"]
            command += ["--bucket_internal"]

    if "starcoder2" in model_name.lower():
        command += ["--flash_attention_recompute"]

    if "gemma" in model_name.lower():
        command += ["--use_flash_attention"]

    if "decilm" in model_name.lower():
        command += ["--sdp_on_bf16"]

    if "mamba-130m-hf" in model_name.lower():
        command += ["--sdp_on_bf16"]

    if (reuse_cache or torch_compile) and not parallel_strategy == "tp" and not is_starcoder_first_gen_model:
        command += ["--reuse_cache"]

    if torch_compile:
        command += ["--torch_compile"]
        if parallel_strategy == "tp":
            command += ["--use_flash_attention"]
            command += ["--flash_attention_recompute"]
        env_variables["PT_ENABLE_INT64_SUPPORT"] = "1"
        env_variables["PT_HPU_LAZY_MODE"] = "0"
    else:
        command += [
            "--use_hpu_graphs",
        ]

    if not deepspeed:
        command.append("--bf16")

    if contrastive_search:
        command += ["--top_k 4", "--penalty_alpha 0.5"]

    if num_beams > 1:
        command += [
            f"--num_beams {num_beams}",
            "--bucket_internal --bucket_size 64",
        ]

    if num_return_sequences > 1:
        command += [
            f"--num_return_sequences {num_return_sequences}",
        ]

    if fp8:
        if "--trim_logits" not in command:
            command += ["--trim_logits"]
        if "Llama-2" in model_name:
            command.insert(-2, "--use_flash_attention")
            command.insert(-2, "--flash_attention_recompute")
            command.insert(-2, "--bucket_size 128")
            command.insert(-2, "--bucket_internal")
        if "Mistral" in model_name:
            command.insert(-2, "--use_flash_attention")
            command.insert(-2, "--flash_attention_recompute")
            command.insert(-2, "--attn_softmax_bf16")
            command.insert(-2, "--trim_logits")
        if "Mixtral" in model_name:
            command.insert(-2, "--bucket_size 128")
            command.insert(-2, "--bucket_internal")
        elif "falcon-180b" in model_name.lower():
            command.insert(-2, "--flash_attention_recompute")

        global prev_quant_model_name
        global prev_quant_rank
        measure_command = None
        # FP8 Measurement only needed
        if (prev_quant_model_name is None) or (prev_quant_model_name != model_name) or (prev_quant_rank != world_size):
            measure_command = [
                x for x in command if not x.startswith("--max_new_tokens")
            ]  # Remove max_new_tokens for measurement
            measure_command = [
                x if not x.startswith("--batch_size") else "--batch_size 1" for x in measure_command
            ]  # Remove batch_size for measurement

            prev_quant_model_name = model_name
            prev_quant_rank = world_size

        # FP8 text generation
        command += [
            f"--max_input_tokens {max_input_tokens}",
            "--limit_hpu_graphs",
        ]
    if load_quantized_model_with_autogptq:
        command += ["--load_quantized_model_with_autogptq"]
    if load_quantized_model_with_autoawq:
        command += ["--load_quantized_model_with_autoawq"]
    if parallel_strategy is not None:
        command += [
            f"--parallel_strategy={parallel_strategy}",
        ]

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        command.append(f"--token {token.value}")

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")

        if fp8:
            env_variables["TQDM_DISABLE"] = "1"
            if measure_command is not None:
                measure_command.append(f"--token {token.value}")
                env_variables["QUANT_CONFIG"] = os.path.join(
                    path_to_example_dir, "text-generation/quantization_config/maxabs_measure_include_outputs.json"
                )
                measure_command = [x for y in measure_command for x in re.split(pattern, y) if x]
                print(f"\n\nMeasure Command to test: {' '.join(measure_command[:-2])}\n")
                proc = subprocess.run(measure_command, env=env_variables)

                # Ensure the run finished without any issue
                # Use try-except to avoid logging the token if used
                try:
                    assert proc.returncode == 0
                except AssertionError as e:
                    if "'--token', 'hf_" in e.args[0]:
                        e.args = (f"The following command failed:\n{' '.join(measure_command[:-2])}",)
                    raise

            if "Mixtral" in model_name:
                env_variables["QUANT_CONFIG"] = os.path.join(
                    path_to_example_dir, "text-generation/quantization_config/maxabs_quant_mixtral.json"
                )
            elif "falcon-180b" in model_name.lower():
                env_variables["PT_HPU_DISABLE_ASYNC_COLLECTIVE"] = "1"
                env_variables["QUANT_CONFIG"] = os.path.join(
                    path_to_example_dir, "text-generation/quantization_config/maxabs_quant.json"
                )
            else:
                env_variables["QUANT_CONFIG"] = os.path.join(
                    path_to_example_dir, "text-generation/quantization_config/maxabs_quant.json"
                )

        command = [x for y in command for x in re.split(pattern, y) if x]
        if "starcoder" in model_name and check_output:
            command.append("--prompt")
            command.append("def print_hello_world():")

        set_seed(42)

        print(f"\n\nCommand to test: {' '.join(command)}\n")
        proc = subprocess.run(command, env=env_variables)

        # Ensure the run finished without any issue
        # Use try-except to avoid logging the token if used
        try:
            assert proc.returncode == 0
        except AssertionError as e:
            if "'--token', 'hf_" in e.args[0]:
                e.args = (f"The following command failed:\n{' '.join(command[:-2])}",)
            raise

        with open(Path(tmp_dir) / "results.json") as fp:
            results = json.load(fp)

        # Ensure performance requirements (throughput) are met
        baseline.assertRef(
            compare=lambda actual, ref: actual >= (2 - TIME_PERF_FACTOR) * ref,
            context=[OH_DEVICE_CONTEXT],
            throughput=results["throughput"],
        )

        # Verify output for 1 HPU, BF16
        if check_output:
            baseline.assertRef(
                compare=operator.eq,
                context=[OH_DEVICE_CONTEXT],
                output=results["output"][0][0],
            )


@pytest.mark.parametrize("model_name, batch_size, reuse_cache, check_output", MODELS_TO_TEST["bf16_1x"])
def test_text_generation_bf16_1x(
    model_name: str, batch_size: int, reuse_cache: bool, check_output: bool, baseline, token
):
    _test_text_generation(
        model_name=model_name,
        baseline=baseline,
        token=token,
        batch_size=batch_size,
        reuse_cache=reuse_cache,
        check_output=check_output,
    )


@pytest.mark.skipif(condition=bool("gaudi1" == OH_DEVICE_CONTEXT), reason=f"Skipping test for {OH_DEVICE_CONTEXT}")
@pytest.mark.parametrize(
    "model_name, world_size, batch_size, reuse_cache, input_len, output_len", MODELS_TO_TEST["fp8"]
)
def test_text_generation_fp8(
    model_name: str,
    world_size: int,
    batch_size: int,
    reuse_cache: bool,
    input_len: int,
    output_len: int,
    baseline,
    token,
):
    deepspeed = True if world_size > 1 else False
    _test_text_generation(
        model_name,
        baseline,
        token,
        deepspeed=deepspeed,
        world_size=world_size,
        fp8=True,
        batch_size=batch_size,
        reuse_cache=reuse_cache,
        max_input_tokens=input_len,
        max_output_tokens=output_len,
    )


@pytest.mark.skipif(condition=bool("gaudi1" == OH_DEVICE_CONTEXT), reason=f"Skipping test for {OH_DEVICE_CONTEXT}")
@pytest.mark.parametrize(
    "model_name, world_size, batch_size, reuse_cache, input_len, output_len",
    MODELS_TO_TEST["load_quantized_model_with_autogptq"],
)
def test_text_generation_gptq(
    model_name: str,
    world_size: int,
    batch_size: int,
    reuse_cache: bool,
    input_len: int,
    output_len: int,
    baseline,
    token,
):
    deepspeed = True if world_size > 1 else False
    _test_text_generation(
        model_name,
        baseline,
        token,
        deepspeed=deepspeed,
        world_size=world_size,
        fp8=False,
        load_quantized_model_with_autogptq=True,
        batch_size=batch_size,
        reuse_cache=reuse_cache,
        max_input_tokens=input_len,
        max_output_tokens=output_len,
    )


@pytest.mark.skipif(condition=bool("gaudi1" == OH_DEVICE_CONTEXT), reason=f"Skipping test for {OH_DEVICE_CONTEXT}")
@pytest.mark.parametrize(
    "model_name, world_size, batch_size, reuse_cache, input_len, output_len",
    MODELS_TO_TEST["load_quantized_model_with_autoawq"],
)
def test_text_generation_awq(
    model_name: str,
    world_size: int,
    batch_size: int,
    reuse_cache: bool,
    input_len: int,
    output_len: int,
    baseline,
    token,
):
    deepspeed = True if world_size > 1 else False
    _test_text_generation(
        model_name,
        baseline,
        token,
        deepspeed=deepspeed,
        world_size=world_size,
        fp8=False,
        load_quantized_model_with_autoawq=True,
        batch_size=batch_size,
        reuse_cache=reuse_cache,
        max_input_tokens=input_len,
        max_output_tokens=output_len,
    )


@pytest.mark.parametrize("model_name, world_size, batch_size", MODELS_TO_TEST["deepspeed"])
def test_text_generation_deepspeed(model_name: str, world_size: int, batch_size: int, baseline, token):
    _test_text_generation(model_name, baseline, token, deepspeed=True, world_size=world_size, batch_size=batch_size)


@pytest.mark.skipif(condition=bool("gaudi1" == OH_DEVICE_CONTEXT), reason=f"Skipping test for {OH_DEVICE_CONTEXT}")
@pytest.mark.parametrize("model_name", MODELS_TO_TEST["torch_compile"])
def test_text_generation_torch_compile(model_name: str, baseline, token):
    _test_text_generation(model_name, baseline, token, torch_compile=True)


@pytest.mark.skipif(condition=bool("gaudi1" == OH_DEVICE_CONTEXT), reason=f"Skipping test for {OH_DEVICE_CONTEXT}")
@pytest.mark.parametrize("model_name", MODELS_TO_TEST["torch_compile_distributed"])
def test_text_generation_torch_compile_distributed(model_name: str, baseline, token):
    world_size = 8
    _test_text_generation(model_name, baseline, token, deepspeed=True, world_size=world_size, torch_compile=True)


@pytest.mark.skipif(condition=bool("gaudi1" == OH_DEVICE_CONTEXT), reason=f"Skipping test for {OH_DEVICE_CONTEXT}")
@pytest.mark.parametrize("model_name", MODELS_TO_TEST["distributed_tp"])
def test_text_generation_distributed_tp(model_name: str, baseline, token):
    world_size = 8
    _test_text_generation(
        model_name,
        baseline,
        token,
        batch_size=64,
        max_input_tokens=128,
        world_size=world_size,
        torch_compile=True,
        parallel_strategy="tp",
    )


@pytest.mark.parametrize("model_name, batch_size, reuse_cache", MODELS_TO_TEST["contrastive_search"])
def test_text_generation_contrastive_search(model_name: str, batch_size: int, reuse_cache: bool, baseline, token):
    _test_text_generation(model_name, baseline, token, batch_size, reuse_cache, contrastive_search=True)


@pytest.mark.skipif(condition=bool("gaudi1" == OH_DEVICE_CONTEXT), reason=f"Skipping test for {OH_DEVICE_CONTEXT}")
@pytest.mark.parametrize("model_name, batch_size, reuse_cache", MODELS_TO_TEST["beam_search"])
def test_text_generation_beam_search(model_name: str, batch_size: int, reuse_cache: bool, baseline, token):
    _test_text_generation(model_name, baseline, token, batch_size, reuse_cache, num_beams=3)
    _test_text_generation(model_name, baseline, token, batch_size, reuse_cache, num_beams=3, num_return_sequences=2)


class TextGenPipeline(TestCase):
    def test_text_generation_pipeline_script(self):
        path_to_script = (
            Path(os.path.dirname(__file__)).parent
            / "examples"
            / "text-generation"
            / "text-generation-pipeline"
            / "run_pipeline.py"
        )

        cmd_line = f"""ls {path_to_script}""".split()

        # check find existence
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()

        # Ensure the run finished without any issue
        self.assertEqual(return_code, 0)

    def test_text_generation_pipeline_falcon(self):
        path_to_script = (
            Path(os.path.dirname(__file__)).parent
            / "examples"
            / "text-generation"
            / "text-generation-pipeline"
            / "run_pipeline.py"
        )
        sys.path.append((Path(os.path.dirname(__file__)).parent / "examples" / "text-generation"))
        cmd_line = f"""
                 python3
                 {path_to_script}
                 --model_name_or_path tiiuae/falcon-7b
                 --max_new_tokens 100
                 --bf16
                 --use_hpu_graphs
                 --use_kv_cache
                 --do_sample
                 """.split()
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()

        # Ensure the run finished without any issue
        self.assertEqual(return_code, 0)
