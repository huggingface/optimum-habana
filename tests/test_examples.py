# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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
import re
import subprocess
from distutils.util import strtobool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Tuple, Union
from unittest import TestCase

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
    MODEL_MAPPING,
)
from transformers.testing_utils import slow

from .utils import (
    MODELS_TO_TEST_FOR_AUDIO_CLASSIFICATION,
    MODELS_TO_TEST_FOR_CAUSAL_LANGUAGE_MODELING,
    MODELS_TO_TEST_FOR_IMAGE_CLASSIFICATION,
    MODELS_TO_TEST_FOR_IMAGE_TEXT,
    MODELS_TO_TEST_FOR_MASKED_LANGUAGE_MODELING,
    MODELS_TO_TEST_FOR_QUESTION_ANSWERING,
    MODELS_TO_TEST_FOR_SEQ2SEQ,
    MODELS_TO_TEST_FOR_SEQUENCE_CLASSIFICATION,
    MODELS_TO_TEST_FOR_SPEECH_RECOGNITION,
    MODELS_TO_TEST_MAPPING,
    OH_DEVICE_CONTEXT,
)


BASELINE_DIRECTORY = Path(__file__).parent.resolve() / Path("baselines")
# Models should reach at least 99% of their baseline accuracy
ACCURACY_PERF_FACTOR = 0.99
# Trainings/Evaluations should last at most 5% longer than the baseline
TIME_PERF_FACTOR = 1.05


IS_GAUDI2 = bool("gaudi2" == OH_DEVICE_CONTEXT)


def _get_supported_models_for_script(
    models_to_test: Dict[str, List[Tuple[str]]],
    task_mapping: Dict[str, str],
    valid_models_for_task: List[str],
) -> List[Tuple[str]]:
    """
    Filter models that can perform the task from models_to_test.
    Args:
        models_to_test: mapping between a model type and a tuple (model_name_or_path, gaudi_config_name).
        task_mapping: mapping between a model config and a model class.
        valid_models_for_task: list of models to test for a specific task.
    Returns:
        A list of models that are supported for the task.
        Each element of the list follows the same format: (model_type, (model_name_or_path, gaudi_config_name)).
    """

    def is_valid_model_type(model_type: str) -> bool:
        true_model_type = "llama" if model_type == "llama_guard" else model_type
        if model_type in ("protst", "chatglm"):
            in_task_mapping = True
        else:
            # llama_guard is not a model type in Transformers so CONFIG_MAPPING wouldn't find it
            in_task_mapping = CONFIG_MAPPING[true_model_type] in task_mapping
        in_valid_models_for_task = model_type in valid_models_for_task
        if in_task_mapping and in_valid_models_for_task:
            return True
        return False

    return [
        model for model_type, models in models_to_test.items() for model in models if is_valid_model_type(model_type)
    ]


_SCRIPT_TO_MODEL_MAPPING = {
    "run_qa": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODELS_TO_TEST_FOR_QUESTION_ANSWERING,
    ),
    "run_glue": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODELS_TO_TEST_FOR_SEQUENCE_CLASSIFICATION,
    ),
    "run_clm": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODELS_TO_TEST_FOR_CAUSAL_LANGUAGE_MODELING,
    ),
    "run_summarization": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODELS_TO_TEST_FOR_SEQ2SEQ,
    ),
    "run_image_classification": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
        MODELS_TO_TEST_FOR_IMAGE_CLASSIFICATION,
    ),
    "run_mlm": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODELS_TO_TEST_FOR_MASKED_LANGUAGE_MODELING,
    ),
    "run_audio_classification": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
        MODELS_TO_TEST_FOR_AUDIO_CLASSIFICATION,
    ),
    "run_speech_recognition_ctc": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_CTC_MAPPING,
        MODELS_TO_TEST_FOR_SPEECH_RECOGNITION,
    ),
    "run_seq2seq_qa": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODELS_TO_TEST_FOR_SEQ2SEQ,
    ),
    "run_clip": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_MAPPING,
        MODELS_TO_TEST_FOR_IMAGE_TEXT,
    ),
    "run_bridgetower": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_MAPPING,
        ["bridgetower"],
    ),
    "run_lora_clm": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        ["llama", "falcon"],
    ),
    "run_speech_recognition_seq2seq": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
        MODELS_TO_TEST_FOR_SPEECH_RECOGNITION,
    ),
    "sft": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        ["llama", "qwen2"],
    ),
    "dpo": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        ["llama"],
    ),
    "reward_modeling": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        ["llama"],
    ),
    "ppo": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        ["llama"],
    ),
    "run_prompt_tuning_clm": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        ["llama"],
    ),
    "run_sequence_classification": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_MAPPING,
        ["protst"],
    ),
    "run_multitask_prompt_tuning": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        ["t5"],
    ),
    "peft_poly_seq2seq_with_generate": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        ["t5"],
    ),
    "run_image2text_lora_finetune": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_VISION_2_SEQ_MAPPING,
        ["idefics2", "mllama", "llava"],
    ),
}


class ExampleTestMeta(type):
    """
    Metaclass that takes care of creating the proper example tests for a given task.
    It uses example_name to figure out which models support this task, and create a run example test for each of these
    models.
    """

    @staticmethod
    def to_test(
        model_name: str,
        multi_card: bool,
        deepspeed: bool,
        example_name: str,
        fsdp: bool,
        fp8: bool,
        eager_mode: bool,
        task_name: str,
    ):
        models_with_specific_rules = [
            "albert-xxlarge-v1",
            "gpt2-xl",
            "facebook/wav2vec2-base",
            "facebook/wav2vec2-large-lv60",
            "BridgeTower/bridgetower-large-itm-mlm-itc",
            "EleutherAI/gpt-neox-20b",
            "google/flan-t5-xxl",
            "tiiuae/falcon-40b",
            "bigscience/bloom-7b1",
            "codellama/CodeLlama-13b-Instruct-hf",
            "MIT/ast-finetuned-speech-commands-v2",
            "meta-llama/LlamaGuard-7b",
            "THUDM/chatglm3-6b",
        ]

        case_only_in_gaudi2 = [
            "sft",
            "dpo",
            "reward_modeling",
            "ppo",
            "prompt_tuning",
            "peft_poly",
            "run_sequence_classification",
            "run_image2text_lora_finetune",
        ]

        models_measured_on_eager_mode = ["google/gemma-2b-it"]

        if (fsdp or fp8) and not IS_GAUDI2:
            return False
        elif (
            any(case in example_name for case in case_only_in_gaudi2)
            or task_name in ("llama-adapter", "vera", "ia3", "adalora", "ln_tuning", "mamamiya405/finred")
        ) and not IS_GAUDI2:
            return False
        elif "Qwen2-72B" in model_name and task_name != "trl-sft-qwen":
            return False
        elif "llama" in model_name and "trl-sft-chat" in task_name:
            return False
        elif ("qwen2" in model_name or "Qwen2" in model_name) and task_name == "trl-sft":
            return False
        elif "llama" in model_name and "trl-sft-qwen" in task_name:
            return False
        elif "Llama-3.1-8B" in model_name:
            if multi_card:
                return False
            elif task_name == "tatsu-lab/alpaca":
                return True
        elif "falcon" in model_name and task_name in (
            "llama-adapter",
            "databricks/databricks-dolly-15k",
            "vera",
            "ia3",
            "adalora",
            "ln_tuning",
            "tatsu-lab/alpaca_cp",
        ):
            return False
        elif eager_mode and model_name not in models_measured_on_eager_mode:
            return False
        elif "gemma" in model_name and not IS_GAUDI2:
            return False
        elif model_name not in models_with_specific_rules and not deepspeed:
            return True
        elif model_name == "gpt2-xl" and deepspeed:
            # GPT2-XL is tested only with DeepSpeed
            return True
        elif "gpt-neox" in model_name and IS_GAUDI2 and deepspeed:
            # GPT-NeoX is tested only on Gaudi2 and with DeepSpeed
            return True
        elif "flan-t5" in model_name and IS_GAUDI2 and deepspeed:
            # Flan-T5 is tested only on Gaudi2 and with DeepSpeed
            return True
        elif "CodeLlama" in model_name and IS_GAUDI2 and deepspeed:
            # CodeLlama is tested only on Gaudi2 and with DeepSpeed
            return True
        elif "Qwen2-72B" in model_name and IS_GAUDI2 and deepspeed:
            return True
        elif model_name == "albert-xxlarge-v1":
            if (("RUN_ALBERT_XXL_1X" in os.environ) and strtobool(os.environ["RUN_ALBERT_XXL_1X"])) or multi_card:
                # ALBERT XXL 1X is tested only if the required flag is present because it takes long
                return True
        elif "wav2vec2-base" in model_name and example_name == "run_audio_classification":
            return True
        elif "wav2vec2-large" in model_name and example_name == "run_speech_recognition_ctc":
            return True
        elif "bridgetower" in model_name and IS_GAUDI2:
            return True
        elif "falcon" in model_name and IS_GAUDI2 and not fsdp and not fp8:
            return True
        elif "bloom" in model_name and deepspeed and not IS_GAUDI2:
            return True
        elif "LlamaGuard" in model_name and deepspeed and IS_GAUDI2:
            return True
        elif "ast-finetuned-speech-commands-v2" in model_name and IS_GAUDI2:
            return True
        elif "huggyllama" in model_name and IS_GAUDI2 and deepspeed:
            return True
        elif "gemma" in model_name and IS_GAUDI2:
            return True
        elif "chatglm3" in model_name and IS_GAUDI2 and deepspeed:
            return True

        return False

    def __new__(
        cls,
        name,
        bases,
        attrs,
        example_name=None,
        multi_card=False,
        deepspeed=False,
        fsdp=False,
        torch_compile=False,
        fp8=False,
        eager_mode=False,
        compile_dynamic: Optional[bool] = None,
    ):
        distribution = "single_card"
        if multi_card:
            distribution = "multi_card"
        elif deepspeed:
            distribution = "deepspeed"
        if example_name is not None:
            models_to_test = _SCRIPT_TO_MODEL_MAPPING.get(example_name)
            if models_to_test is None:
                if example_name in ["run_esmfold", "run_lora_clm", "run_zero_shot_eval"]:
                    attrs[f"test_{example_name}_{distribution}"] = cls._create_test(None, None, None, None, None)
                    attrs["EXAMPLE_NAME"] = example_name
                    return super().__new__(cls, name, bases, attrs)
                else:
                    raise AttributeError(
                        f"Could not create class because no model was found for example {example_name}"
                    )

        for model_name, gaudi_config_name in models_to_test:
            if cls.to_test(model_name, multi_card, deepspeed, example_name, fsdp, fp8, eager_mode, attrs["TASK_NAME"]):
                attrs[f"test_{example_name}_{model_name.split('/')[-1]}_{distribution}"] = cls._create_test(
                    model_name, gaudi_config_name, multi_card, deepspeed, fsdp, torch_compile, fp8
                )

        attrs["EXAMPLE_NAME"] = example_name
        return super().__new__(cls, name, bases, attrs)

    @classmethod
    def _create_test(
        cls,
        model_name: str,
        gaudi_config_name: str,
        multi_card: bool = False,
        deepspeed: bool = False,
        fsdp: bool = False,
        torch_compile: bool = False,
        fp8: bool = False,
        compile_dynamic: Optional[bool] = None,
    ) -> Callable[[], None]:
        """
        Create a test function that runs an example for a specific (model_name, gaudi_config_name) pair.
        Args:
            model_name (str): the model_name_or_path.
            gaudi_config_name (str): the gaudi config name.
            multi_card (bool): whether it is a distributed run or not.
            deepspeed (bool): whether deepspeed should be used or not.
        Returns:
            The test function that runs the example.
        """

        @slow
        def test(self):
            if self.EXAMPLE_NAME is None:
                raise ValueError("An example name must be provided")
            example_script = Path(self.EXAMPLE_DIR).glob(f"*/{self.EXAMPLE_NAME}.py")
            example_script = list(example_script)
            if len(example_script) == 0:
                raise RuntimeError(f"Could not find {self.EXAMPLE_NAME}.py in examples located in {self.EXAMPLE_DIR}")
            elif len(example_script) > 1:
                raise RuntimeError(f"Found more than {self.EXAMPLE_NAME}.py in examples located in {self.EXAMPLE_DIR}")
            else:
                example_script = example_script[0]

            # The ESMFold example has no arguments, so we can execute it right away
            if self.EXAMPLE_NAME == "run_esmfold":
                cmd_line = f"""
                        python3
                        {example_script}
                        """.split()
                print(f"\n\nCommand to test: {' '.join(cmd_line[:])}\n")
                p = subprocess.Popen(cmd_line)
                return_code = p.wait()
                # Ensure the run finished without any issue
                self.assertEqual(return_code, 0)
                return
            elif self.EXAMPLE_NAME == "run_zero_shot_eval":
                with TemporaryDirectory() as tmp_dir:
                    cmd_line = f"""
                        python3
                        {example_script}
                        --output_dir {tmp_dir}
                        --bf16
                        --max_seq_length 1024
                    """.split()
                    print(f"\n\nCommand to test: {' '.join(cmd_line[:])}\n")
                    p = subprocess.Popen(cmd_line)
                    return_code = p.wait()
                    # Ensure the run finished without any issue
                    self.assertEqual(return_code, 0)
                    # Assess accuracy
                    with open(Path(tmp_dir) / "accuracy_metrics.json") as fp:
                        results = json.load(fp)
                        baseline = 0.43 if IS_GAUDI2 else 0.42
                        self.assertGreaterEqual(results["accuracy"], baseline)
                return
            elif self.EXAMPLE_NAME == "run_clip":
                if os.environ.get("DATA_CACHE", None) is None:
                    from .clip_coco_utils import COCO_URLS, download_files

                    download_files(COCO_URLS)
                from .clip_coco_utils import create_clip_roberta_model

                create_clip_roberta_model()

            self._install_requirements(example_script.parent / "requirements.txt")

            # collect baseline from <model_name>_eager.json if eager_mode is True
            if self.EAGER_MODE:
                baseline_name = model_name.split("/")[-1].replace("-", "_").replace(".", "_") + "_eager"
            else:
                baseline_name = model_name.split("/")[-1].replace("-", "_").replace(".", "_")

            path_to_baseline = BASELINE_DIRECTORY / Path(baseline_name).with_suffix(".json")

            with path_to_baseline.open("r") as json_file:
                device = "gaudi2" if IS_GAUDI2 else "gaudi"
                baseline = json.load(json_file)[device]
                if isinstance(self.TASK_NAME, list):
                    for key in self.TASK_NAME:
                        if key in baseline:
                            baseline = baseline[key]
                            break
                    if "num_train_epochs" not in baseline:
                        raise ValueError(
                            f"Couldn't find a baseline associated to any of these tasks: {self.TASK_NAME}."
                        )
                    self.TASK_NAME = key
                else:
                    baseline = baseline[self.TASK_NAME]

            distribution = "single_card"
            if multi_card:
                distribution = "multi_card"
            elif deepspeed:
                distribution = "deepspeed"

            env_variables = os.environ.copy()
            if "falcon" in model_name:
                env_variables["PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST"] = str(example_script.parent / "ops_bf16.txt")
            elif "flan" in model_name:
                env_variables["PT_HPU_MAX_COMPOUND_OP_SIZE"] = "512"
            elif "bloom" in model_name:
                env_variables["DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED"] = "1"
                env_variables["PT_HPU_MAX_COMPOUND_OP_SYNC"] = "1"
                env_variables["PT_HPU_MAX_COMPOUND_OP_SIZE"] = "1"
            elif "Qwen2-72B" in model_name:
                env_variables["DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED"] = "1"
            elif fsdp:
                if "llama" in model_name:
                    env_variables["PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST"] = str(
                        example_script.parent / "ops_bf16.txt"
                    )
                env_variables["PT_HPU_LAZY_MODE"] = "0"
            elif deepspeed and "gpt-neox-20b" in model_name:
                env_variables["LD_PRELOAD"] = ""

            if fp8 and "llama" in model_name:
                env_variables["PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST"] = str(example_script.parent / "ops_bf16.txt")

            extra_command_line_arguments = baseline.get("distribution").get(distribution).get("extra_arguments", [])

            if self.EAGER_MODE:
                env_variables["PT_HPU_LAZY_MODE"] = "0"
                if "--use_hpu_graphs_for_inference" in extra_command_line_arguments:
                    extra_command_line_arguments.remove("--use_hpu_graphs_for_inference")
            if os.environ.get("DATA_CACHE", None) is not None and self.EXAMPLE_NAME == "run_clip":
                extra_command_line_arguments[0] = "--data_dir {}".format(os.environ["DATA_CACHE"])

            if torch_compile and (
                model_name == "bert-large-uncased-whole-word-masking"
                or model_name == "roberta-large"
                or model_name == "albert-xxlarge-v1"
                or model_name == "./clip-roberta"
            ):
                extra_command_line_arguments.append("--torch_compile_backend hpu_backend")
                extra_command_line_arguments.append("--torch_compile")
                if compile_dynamic is not None:
                    extra_command_line_arguments.append(f"--compile_dynamic {compile_dynamic}")
                if "--use_hpu_graphs_for_inference" in extra_command_line_arguments:
                    extra_command_line_arguments.remove("--use_hpu_graphs_for_inference")
                env_variables["PT_HPU_LAZY_MODE"] = "0"
                env_variables["PT_ENABLE_INT64_SUPPORT"] = "1"

            if self.EXAMPLE_NAME == "run_audio_classification":
                extra_command_line_arguments.append("--sdp_on_bf16")

            if self.EXAMPLE_NAME == "run_image_classification":
                extra_command_line_arguments.append("--sdp_on_bf16")

            if self.EXAMPLE_NAME == "run_glue":
                if model_name == "bert-large-uncased-whole-word-masking":
                    extra_command_line_arguments.append("--sdp_on_bf16")

            if self.EXAMPLE_NAME == "run_qa":
                if model_name == "bert-large-uncased-whole-word-masking" or model_name == "albert-large-v2":
                    extra_command_line_arguments.append("--sdp_on_bf16")

            if self.EXAMPLE_NAME == "run_bridgetower":
                if model_name == "BridgeTower/bridgetower-large-itm-mlm-itc":
                    extra_command_line_arguments.append("--sdp_on_bf16")

            if self.EXAMPLE_NAME == "run_speech_recognition_seq2seq":
                if model_name == "openai/whisper-small":
                    extra_command_line_arguments.append("--sdp_on_bf16")

            if self.EXAMPLE_NAME == "run_clip":
                extra_command_line_arguments.append("--sdp_on_bf16")

            if self.EXAMPLE_NAME == "run_image2text_lora_finetune":
                extra_command_line_arguments.append("--sdp_on_bf16")

            with TemporaryDirectory() as tmp_dir:
                cmd_line = self._create_command_line(
                    multi_card,
                    deepspeed,
                    fsdp,
                    example_script,
                    model_name,
                    gaudi_config_name,
                    tmp_dir,
                    task=self.TASK_NAME,
                    lr=baseline.get("distribution").get(distribution).get("learning_rate"),
                    train_batch_size=baseline.get("distribution").get(distribution).get("train_batch_size"),
                    eval_batch_size=baseline.get("eval_batch_size"),
                    num_epochs=baseline.get("num_train_epochs"),
                    extra_command_line_arguments=extra_command_line_arguments,
                )
                print(f"\n\nCommand to test: {' '.join(cmd_line[:])}\n")
                p = subprocess.Popen(cmd_line, env=env_variables)
                return_code = p.wait()

                # Ensure the run finished without any issue
                self.assertEqual(return_code, 0)

                with open(Path(tmp_dir) / "all_results.json") as fp:
                    results = json.load(fp)
                # Ensure performance requirements (accuracy, training time) are met
                self.assert_no_regression(results, baseline.get("distribution").get(distribution), model_name)

            # TODO: is a cleanup of the dataset cache needed?
            # self._cleanup_dataset_cache()

        return test


class ExampleTesterBase(TestCase):
    """
    Base example tester class.
    Attributes:
        EXAMPLE_DIR (`str` or `os.Pathlike`): the directory containing the examples.
        EXAMPLE_NAME (`str`): the name of the example script without the file extension, e.g. run_qa, run_glue, etc.
        TASK_NAME (`str`): the name of the dataset to use.
        DATASET_PARAMETER_NAME (`str`): the argument name to use for the dataset parameter.
            Most of the time it will be "dataset_name", but for some tasks on a benchmark it might be something else.
        MAX_SEQ_LENGTH ('str'): the max_seq_length argument for this dataset.
            The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
    """

    EXAMPLE_DIR = Path(os.path.dirname(__file__)).parent / "examples"
    EXAMPLE_NAME = None
    TASK_NAME = None
    DATASET_PARAMETER_NAME = "dataset_name"
    DATASET_NAME = None
    REGRESSION_METRICS = {
        "eval_f1": (TestCase.assertGreaterEqual, ACCURACY_PERF_FACTOR),
        "eval_accuracy": (TestCase.assertGreaterEqual, ACCURACY_PERF_FACTOR),
        "perplexity": (TestCase.assertLessEqual, 2 - ACCURACY_PERF_FACTOR),
        "eval_rougeLsum": (TestCase.assertGreaterEqual, ACCURACY_PERF_FACTOR),
        "train_runtime": (TestCase.assertLessEqual, TIME_PERF_FACTOR),
        "eval_wer": (TestCase.assertLessEqual, 2 - ACCURACY_PERF_FACTOR),
        "train_samples_per_second": (TestCase.assertGreaterEqual, 2 - TIME_PERF_FACTOR),
        "eval_samples_per_second": (TestCase.assertGreaterEqual, 2 - TIME_PERF_FACTOR),
    }
    EAGER_MODE = False

    def _create_command_line(
        self,
        multi_card: bool,
        deepspeed: bool,
        fsdp: bool,
        script: Path,
        model_name: str,
        gaudi_config_name: str,
        output_dir: str,
        lr: float,
        train_batch_size: int,
        eval_batch_size: int,
        num_epochs: int,
        task: Optional[str] = None,
        extra_command_line_arguments: Optional[List[str]] = None,
    ) -> List[str]:
        dataset_name = self.DATASET_NAME if self.DATASET_NAME is not None else task
        task_option = f"--{self.DATASET_PARAMETER_NAME} {dataset_name}" if task else " "
        if task in ["multitask-prompt-tuning", "poly-tuning"]:
            task_option = " "
        cmd_line = ["python3"]
        if multi_card:
            cmd_line.append(f"{script.parent.parent / 'gaudi_spawn.py'}")
            cmd_line.append("--world_size 8")
            cmd_line.append("--use_mpi")
        elif deepspeed:
            cmd_line = [
                "deepspeed",
                "--num_nodes 1",
                "--num_gpus 8",
                "--no_local_rank",
            ]
        if self.EXAMPLE_NAME in ["dpo", "reward_modeling"]:
            cmd_line += [
                f"{script}",
                f"--model_name_or_path {model_name}",
                f"--tokenizer_name_or_path {model_name}",
                f"--output_dir {output_dir}",
                f"--per_device_train_batch_size {train_batch_size}",
                f"--per_device_eval_batch_size {eval_batch_size}",
            ]
        elif self.EXAMPLE_NAME == "ppo":
            cmd_line += [
                f"{script}",
                f"--model_name_or_path {model_name}",
                f"--tokenizer_name_or_path {model_name}",
                f"--output_dir {output_dir}",
                f"--batch_size {train_batch_size}",
            ]
        else:
            cmd_line += [
                f"{script}",
                f"--model_name_or_path {model_name}",
                f"--gaudi_config_name {gaudi_config_name}",
                f"{task_option}",
                "--do_train",
                f"--output_dir {output_dir}",
                "--overwrite_output_dir",
                f"--learning_rate {lr}",
                f"--per_device_train_batch_size {train_batch_size}",
                f"--per_device_eval_batch_size {eval_batch_size}",
                f" --num_train_epochs {num_epochs}",
                "--use_habana",
                "--throughput_warmup_steps 3",
                "--save_strategy no",
            ]

        if "compile" in task or "--torch_compile" in extra_command_line_arguments:
            cmd_line += ["--use_lazy_mode False"]
        elif self.EXAMPLE_NAME not in ["dpo", "ppo", "reward_modeling"]:
            cmd_line += ["--use_lazy_mode"]

        if "bloom" not in model_name and self.EXAMPLE_NAME not in ["dpo", "ppo", "reward_modeling"]:
            cmd_line.append("--do_eval")

        if extra_command_line_arguments is not None:
            cmd_line += extra_command_line_arguments

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
        return [x for y in cmd_line for x in re.split(pattern, y) if x]

    def _install_requirements(self, requirements_filename: Union[str, os.PathLike]):
        """
        Installs the necessary requirements to run the example if the provided file exists, otherwise does nothing.
        """

        if not Path(requirements_filename).exists():
            return

        cmd_line = f"pip install -r {requirements_filename}".split()
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()
        self.assertEqual(return_code, 0)

    def assert_no_regression(self, results: Dict, baseline: Dict, model_name: str):
        """
        Assert whether all possible performance requirements are met.
        Attributes:
            results (Dict): results of the run to assess
            baseline (Dict): baseline to assert whether or not there is regression
        """
        # Gather all the metrics to assess
        metrics_to_assess = []
        for metric_name in self.REGRESSION_METRICS.keys():
            if metric_name in baseline and metric_name in results:
                metrics_to_assess.append(metric_name)
        # There is no accuracy metric for `run_clip.py`, `run_bridgetower.py` and BLOOM
        min_number_metrics = 3
        if (
            self.EXAMPLE_NAME in ["run_clip", "run_bridgetower", "sft", "dpo", "ppo", "reward_modeling"]
            or "bloom" in model_name
        ):
            min_number_metrics = 2

        # Check that at least 3 metrics are assessed:
        # training time + throughput + accuracy metric (F1, accuracy, perplexity,...)
        self.assertGreaterEqual(
            len(metrics_to_assess),
            min_number_metrics,
            (
                f"{len(metrics_to_assess)} asserted metric(s) while at least 3 are expected (throughput + training"
                f" time + accuracy). Metrics to assert: {self.REGRESSION_METRICS.keys()}. Metrics received:"
                f" {baseline.keys()}"
            ),
        )

        # Message to display if one test fails
        # This enables to show all the results and baselines even if one test fails before others
        failure_message = "\n===== Assessed metrics (measured vs thresholded baseline) =====\n"
        for metric_name in metrics_to_assess:
            failure_message += f"{metric_name}: {results[metric_name]} vs {self.REGRESSION_METRICS[metric_name][1] * baseline[metric_name]}\n"

        # Assess metrics
        for metric_name in metrics_to_assess:
            assert_function, threshold_factor = self.REGRESSION_METRICS[metric_name]
            assert_function(
                self,
                results[metric_name],
                threshold_factor * baseline[metric_name],
                msg=f"for metric {metric_name}. {failure_message}",
            )


class TextClassificationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_glue"):
    TASK_NAME = "mrpc"
    DATASET_PARAMETER_NAME = "task_name"


class MultiCardTextClassificationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_glue", multi_card=True
):
    TASK_NAME = "mrpc"
    DATASET_PARAMETER_NAME = "task_name"


class DeepSpeedTextClassificationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_glue", deepspeed=True
):
    TASK_NAME = "mrpc"
    DATASET_PARAMETER_NAME = "task_name"


class QuestionAnsweringExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_qa", torch_compile=True
):
    TASK_NAME = "squad"


class MultiCardQuestionAnsweringExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_qa", multi_card=True, torch_compile=True
):
    TASK_NAME = "squad"


class EagerModeCausalLanguageModelingExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_clm", eager_mode=True
):
    TASK_NAME = "wikitext"
    EAGER_MODE = True


class CausalLanguageModelingExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_clm"):
    TASK_NAME = "wikitext"


class MultiCardCausalLanguageModelingExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_clm", multi_card=True
):
    TASK_NAME = "wikitext"


class DeepspeedCausalLanguageModelingExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_clm", deepspeed=True
):
    TASK_NAME = "wikitext"


class ImageClassificationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_image_classification"
):
    TASK_NAME = "cifar10"


class MultiCardImageClassificationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_image_classification", multi_card=True
):
    TASK_NAME = "cifar10"


class MultiCardMaskedLanguageModelingExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_mlm", multi_card=True
):
    TASK_NAME = "wikitext"


class MultiCardAudioClassificationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_audio_classification", multi_card=True
):
    TASK_NAME = "common_language"


class MultiCardSpeechRecognitionExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_speech_recognition_ctc", multi_card=True
):
    TASK_NAME = "regisss/librispeech_asr_for_optimum_habana_ci"
    DATASET_NAME = os.environ.get("DATA_CACHE", None)


class MultiCardSummarizationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_summarization", multi_card=True
):
    TASK_NAME = "cnn_dailymail"


class MultiCardDynamicCompileSummarizationExampleTester(
    ExampleTesterBase,
    metaclass=ExampleTestMeta,
    example_name="run_summarization",
    multi_card=True,
    compile_dynamic=True,
):
    TASK_NAME = "cnn_dailymail"


class DeepspeedSummarizationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_summarization", deepspeed=True
):
    TASK_NAME = "cnn_dailymail"


class MultiCardSeq2SeqQuestionAnsweringExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_seq2seq_qa", multi_card=True
):
    TASK_NAME = "squad_v2"


class MultiCardVisionLanguageExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_clip", multi_card=True, torch_compile=True
):
    TASK_NAME = "ydshieh/coco_dataset_script"


class ProteinFoldingExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_esmfold"):
    pass


class ProteinFoldingExampleTester2(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_zero_shot_eval"):
    pass


class CausalLanguageModelingLORAExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm"
):
    TASK_NAME = "databricks/databricks-dolly-15k"


class MultiCardCausalLanguageModelingLORAExampleTester2(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm", multi_card=True
):
    TASK_NAME = "mamamiya405/finred"


class MultiCardCausalLanguageModelingLORAExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm", multi_card=True
):
    TASK_NAME = ["tatsu-lab/alpaca", "timdettmers/openassistant-guanaco"]


class MultiCardBridgetowerExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_bridgetower", multi_card=True
):
    TASK_NAME = "jmhessel/newyorker_caption_contest"


class MultiCardSeq2SeqSpeechRecognitionExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_speech_recognition_seq2seq", multi_card=True
):
    TASK_NAME = "mozilla-foundation/common_voice_11_0"


class MultiCardCausalLanguageModelingLORAFSDPCompileExampleTester(
    ExampleTesterBase,
    metaclass=ExampleTestMeta,
    example_name="run_lora_clm",
    multi_card=True,
    fsdp=True,
):
    TASK_NAME = "tatsu-lab/alpaca_fsdpcompile"
    DATASET_NAME = "tatsu-lab/alpaca"


class MultiCardSFTExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="sft", multi_card=True):
    TASK_NAME = "trl-sft"
    DATASET_NAME = "lvwerra/stack-exchange-paired"


class DeepspeedSFTExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="sft", deepspeed=True):
    TASK_NAME = "trl-sft-qwen"
    DATASET_NAME = "philschmid/dolly-15k-oai-style"


class MultiCardSFTChatExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="sft", multi_card=True):
    TASK_NAME = "trl-sft-chat"
    DATASET_NAME = "philschmid/dolly-15k-oai-style"


class MultiCardSFTChatPeftExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="sft", multi_card=True
):
    TASK_NAME = "trl-sft-chat-peft"
    DATASET_NAME = "philschmid/dolly-15k-oai-style"


class MultiCardDPOExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="dpo", multi_card=True):
    TASK_NAME = "trl-dpo"
    DATASET_NAME = "lvwerra/stack-exchange-paired"


class MultiCardRewardExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="reward_modeling", multi_card=True
):
    TASK_NAME = "trl-reward"
    DATASET_NAME = "lvwerra/stack-exchange-paired"


class MultiCardPPOExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="ppo", multi_card=True):
    TASK_NAME = "trl-ppo"
    DATASET_NAME = "lvwerra/stack-exchange-paired"


class MultiCardProteinFoldingClassificationTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_sequence_classification", multi_card=True
):
    TASK_NAME = "prost-sequence-classification"
    DATASET_NAME = "mila-intel/ProtST-BinaryLocalization"


class MultiCardCausalLanguageModelingPromptTuningExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_prompt_tuning_clm", multi_card=True
):
    TASK_NAME = "prompt-tuning"
    DATASET_NAME = "ought/raft"


class MultiCardCausalLanguageModelingPrefixTuningExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_prompt_tuning_clm", multi_card=True
):
    TASK_NAME = "prefix-tuning"
    DATASET_NAME = "ought/raft"


class MultiCardCausalLanguageModelingPTuningExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_prompt_tuning_clm", multi_card=True
):
    TASK_NAME = "p-tuning"
    DATASET_NAME = "ought/raft"


class MultiCardMultiTastPromptPeftExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_multitask_prompt_tuning", multi_card=True
):
    TASK_NAME = "multitask-prompt-tuning"


class MultiCardPolyPeftExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="peft_poly_seq2seq_with_generate", multi_card=True
):
    TASK_NAME = "poly-tuning"


class MultiCardCausalLanguageModelingLlamaAdapterExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm", multi_card=True
):
    TASK_NAME = "llama-adapter"
    DATASET_NAME = "tatsu-lab/alpaca"


class MultiCardCausalLanguageModelingLoRAFP8ExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm", multi_card=True, fp8=True
):
    TASK_NAME = "tatsu-lab/alpaca_fp8"
    DATASET_NAME = "tatsu-lab/alpaca"


class MultiCardImageToTextModelingLoRAExampleTester(
    ExampleTesterBase,
    metaclass=ExampleTestMeta,
    example_name="run_image2text_lora_finetune",
    multi_card=True,
):
    TASK_NAME = "image2text_lora_finetune"
    DATASET_NAME = "nielsr/docvqa_1200_examples"


class MultiCardCausalLanguageModelingVeraExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm", multi_card=True
):
    TASK_NAME = "vera"
    DATASET_NAME = "tatsu-lab/alpaca"


class MultiCardCausalLanguageModelingLnExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm", multi_card=True
):
    TASK_NAME = "ln_tuning"
    DATASET_NAME = "tatsu-lab/alpaca"


class MultiCardCausalLanguageModelingIA3ExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm", multi_card=True
):
    TASK_NAME = "ia3"
    DATASET_NAME = "tatsu-lab/alpaca"


class MultiCardCausalLanguageModelingAdaloraExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm", multi_card=True
):
    TASK_NAME = "adalora"
    DATASET_NAME = "tatsu-lab/alpaca"


class MultiCardCausalLanguageModelingLoRACPExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm", deepspeed=True
):
    TASK_NAME = "tatsu-lab/alpaca_cp"
    DATASET_NAME = "tatsu-lab/alpaca"
