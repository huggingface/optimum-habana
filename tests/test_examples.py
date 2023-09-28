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
    MODELS_TO_TEST_FOR_TEXT_GENERATION,
    MODELS_TO_TEST_MAPPING,
)


BASELINE_DIRECTORY = Path(__file__).parent.resolve() / Path("baselines")
# Models should reach at least 99% of their baseline accuracy
ACCURACY_PERF_FACTOR = 0.99
# Trainings/Evaluations should last at most 5% longer than the baseline
TIME_PERF_FACTOR = 1.05


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
        if model_type == "llama-v2-13b-hf":
            return True
        in_task_mapping = CONFIG_MAPPING[model_type] in task_mapping
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
    "run_generation": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_MAPPING,
        MODELS_TO_TEST_FOR_TEXT_GENERATION,
    ),
}


class ExampleTestMeta(type):
    """
    Metaclass that takes care of creating the proper example tests for a given task.
    It uses example_name to figure out which models support this task, and create a run example test for each of these
    models.
    """

    @staticmethod
    def to_test(model_name: str, multi_card: bool, deepspeed: bool, example_name: str):
        models_with_specific_rules = [
            "albert-xxlarge-v1",
            "gpt2-xl",
            "facebook/wav2vec2-base",
            "facebook/wav2vec2-large-lv60",
        ]

        if model_name not in models_with_specific_rules and not deepspeed:
            return True
        elif model_name == "gpt2-xl" and deepspeed:
            # GPT2-XL is tested only with DeepSpeed
            return True
        elif model_name == "albert-xxlarge-v1":
            if (("RUN_ALBERT_XXL_1X" in os.environ) and strtobool(os.environ["RUN_ALBERT_XXL_1X"])) or multi_card:
                # ALBERT XXL 1X is tested only if the required flag is present because it takes long
                return True
        elif "wav2vec2-base" in model_name and example_name == "run_audio_classification":
            return True
        elif "wav2vec2-large" in model_name and example_name == "run_speech_recognition_ctc":
            return True
        elif "llama-v2-13b-hf" in model_name and example_name == "run_generation":
            return True
        return False

    def __new__(cls, name, bases, attrs, example_name=None, multi_card=False, deepspeed=False):
        distribution = "single_card"
        if multi_card:
            distribution = "multi_card"
        elif deepspeed:
            distribution = "deepspeed"

        if example_name is not None:
            models_to_test = _SCRIPT_TO_MODEL_MAPPING.get(example_name)
            if models_to_test is None:
                if example_name in ["run_esmfold", "run_lora_clm"]:
                    attrs[f"test_{example_name}_{distribution}"] = cls._create_test(None, None, None, None)
                    attrs["EXAMPLE_NAME"] = example_name
                    return super().__new__(cls, name, bases, attrs)
                else:
                    raise AttributeError(
                        f"Could not create class because no model was found for example {example_name}"
                    )
        for model_name, gaudi_config_name in models_to_test:
            if cls.to_test(model_name, multi_card, deepspeed, example_name):
                attrs[f"test_{example_name}_{model_name.split('/')[-1]}_{distribution}"] = cls._create_test(
                    model_name, gaudi_config_name, multi_card, deepspeed
                )
        attrs["EXAMPLE_NAME"] = example_name
        return super().__new__(cls, name, bases, attrs)

    @classmethod
    def _create_test(
        cls, model_name: str, gaudi_config_name: str, multi_card: bool = False, deepspeed: bool = False
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
                p = subprocess.Popen(["python3", example_script])
                return_code = p.wait()

                # Ensure the run finished without any issue
                self.assertEqual(return_code, 0)
                return
            # At the moment, just run the LORA example to check if there is no error
            elif self.EXAMPLE_NAME == "run_lora_clm":
                self._install_requirements(example_script.parent / "requirements.txt")

                command = [
                    "python3",
                    # TODO: uncomment the following lines when LoRA 8x is fixed
                    # f"{example_script.parent.parent / 'gaudi_spawn.py'}",
                    # "--use_mpi",
                    # "--world_size 8",
                    f"{example_script}",
                    "--model_name_or_path huggyllama/llama-7b",
                    "--dataset_name tatsu-lab/alpaca",
                    "--bf16",
                    "--output_dir /tmp/model_lora_llama",
                    "--num_train_epochs 1",
                    "--per_device_train_batch_size 2",
                    "--per_device_eval_batch_size 2",
                    "--gradient_accumulation_steps 4",
                    "--save_strategy no",
                    "--learning_rate 1e-4",
                    "--dataset_concatenation",
                    "--do_train",
                    "--use_habana",
                    "--use_lazy_mode",
                    "--throughput_warmup_steps 3",
                    "--max_steps 100",
                ]
                pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
                command = [x for y in command for x in re.split(pattern, y) if x]
                p = subprocess.Popen(command)
                return_code = p.wait()

                # Ensure the run finished without any issue
                self.assertEqual(return_code, 0)
                return
            # The CLIP example requires COCO and a clip-roberta model
            elif self.EXAMPLE_NAME == "run_clip":
                from .clip_coco_utils import create_clip_roberta_model, download_coco

                download_coco()
                create_clip_roberta_model()
            elif self.EXAMPLE_NAME == "run_generation":
                self._install_requirements(example_script.parent / "requirements.txt")
                command = [
                    "python3",
                    f"{example_script}",
                    "--model_name_or_path /root/czhao/WORK/llama-v2/test-01/13B-hf",
                    "--bf16",
                    "--use_kv_cache",
                    "--use_hpu_graphs",
                    "--max_new_tokens 1024",
                    "--prompt 'While the proposed deep learning framework for CSI feedback based on superimposed coding offers a promising solution to the challenges of massive MIMO and mmWave systems, there are several research areas that can be further explored or improved. In this section, we discuss potential future research directions and enhancements to the current work. Adaptive Compression and Prediction Techniques: One potential improvement to the current deep learning framework is the development of adaptive compression and prediction techniques that can dynamically adjust the level of CSI compression and prediction based on the current channel conditions and system requirements. By incorporating adaptivity into the model, the framework could further optimize the trade-off between feedback overhead, latency, and CSI accuracy. Future research could investigate reinforcement learning or other online learning algorithms to enable such adaptive behavior in the deep learning model. Multi-antenna and Multi-user Superimposed Coding: The current work focuses on single-antenna systems and does not fully explore the potential of superimposed coding in multi-antenna and multi-user scenarios. Future research could extend the proposed framework to handle multiple antennas and users, investigating the impact of superimposed coding on system performance and the interactions between users. This would require the development of novel superimposed coding techniques that consider the spatial dimensions and user interference in multi-antenna and multi-user systems. Robustness to Channel Model Mismatch: The proposed deep learning framework assumes a specific channel model during the training phase. However, in practical wireless communication systems, the actual channel model may deviate from the assumed model. Future research could investigate the robustness of the proposed framework to channel model mismatch and develop techniques to improve its adaptability to different channel models. One potential approach is to incorporate unsupervised or semi-supervised learning methods into the framework, allowing the model to learn from partially labeled or unlabeled data obtained from the actual channel. Integration of Advanced Deep Learning Techniques: The current work employs a conventional encoder-decoder deep learning architecture for CSI feedback. Future research could explore the integration of advanced deep learning techniques, such as attention mechanisms, recurrent neural networks, and graph neural networks, to further improve the performance of the framework. These advanced techniques could enhance the models ability to capture complex channel patterns, temporal dependencies, and spatial correlations, resulting in more accurate CSI feedback and reduced overhead. Joint Optimization of Communication and Sensing: Massive MIMO and mmWave systems can potentially serve dual roles as communication and sensing systems. Future research could investigate the joint optimization of communication and sensing tasks in the context of deep learning-based CSI feedback and superimposed coding. This would involve developing novel architectures and algorithms that can efficiently balance the resource allocation and performance trade-offs between communication and sensing tasks while leveraging the benefits of superimposed coding. Hardware-aware Deep Learning Models: In practical implementations, the performance of deep learning models is affected by the hardware constraints of the devices, such as processing power, memory, and energy consumption. Future research could focus on the development of hardware-aware deep learning models for CSI feedback that consider these constraints during the design and training phases. This would require the investigation of model compression techniques, energy-efficient neural architectures, and quantization methods to enable efficient and low-complexity CSI feedback in resource-constrained devices. Cross-layer Optimization and Co-design: The proposed deep learning framework for CSI feedback primarily focuses on the physical layer of the wireless communication system. Future research could explore cross-layer optimization and co-design, integrating the proposed framework with upper-layer protocols and functionalities, such as Medium Access Control, routing, and Quality of Service. This would involve developing novel algorithms and models that can jointly optimize the performance of the entire communication system, leveraging the benefits of deep learning-based CSI feedback and superimposed coding across multiple layers of the network stack. Federated Learning and Distributed CSI Feedback: In large-scale wireless networks, such as the Internet of Things and 5G, 6G networks, centralized CSI feedback may not be feasible or efficient due to the high overhead and latency requirements. Future research could investigate the application of federated learning and distributed CSI feedback techniques to the proposed deep learning framework. This would involve developing novel algorithms and architectures that allow multiple devices to collaboratively learn and share CSI feedback, reducing the overall overhead and improving the scalability of the system. Security and Privacy Considerations: As deep learning models are integrated into wireless communication systems, security and privacy concerns become increasingly important. Future research could explore potential vulnerabilities in the proposed deep learning framework for CSI feedback, such as adversarial attacks, and develop countermeasures to ensure the integrity and confidentiality of the CSI. This would require the investigation of techniques such as adversarial training, secure multi-party computation, and differential privacy to protect the deep learning models and the underlying communication system. Experimental Validation and Testbed Development: While the current work evaluates the performance of the proposed deep learning framework through simulations and real-world experiments, future research could focus on developing a comprehensive testbed to validate the framework under various realistic scenarios and conditions. This would involve the design and implementation of hardware and software components that can accurately emulate the characteristics of massive MIMO and mmWave systems, as well as the integration of the deep learning framework with existing wireless communication platforms. In conclusion, the deep learning framework for CSI feedback based on superimposed coding presents a promising solution to the challenges of massive MIMO and mmWave systems. However, several research areas can be further explored or improved to enhance the performance, robustness, and applicability of the framework. By investigating these future research directions, the proposed framework can contribute to the development of more efficient, scalable, and secure wireless communication systems that can meet the ever-growing demands of modern applications and services. Integration with Edge and Cloud Computing: With the advent of edge computing and the continued growth of cloud computing, the processing capabilities of wireless communication systems are expanding. Future research could explore the integration of the proposed deep learning framework for CSI feedback with edge and cloud computing paradigms. This would involve'",
                ]
                pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
                command = [x for y in command for x in re.split(pattern, y) if x]
                p = subprocess.Popen(command)
                return_code = p.wait()
                # Ensure the run finished without any issue
                self.assertEqual(return_code, 0)
                return

            self._install_requirements(example_script.parent / "requirements.txt")

            path_to_baseline = BASELINE_DIRECTORY / Path(model_name.split("/")[-1].replace("-", "_")).with_suffix(
                ".json"
            )
            with path_to_baseline.open("r") as json_file:
                baseline = json.load(json_file)[self.TASK_NAME]

            distribution = "single_card"
            if multi_card:
                distribution = "multi_card"
            elif deepspeed:
                distribution = "deepspeed"

            with TemporaryDirectory() as tmp_dir:
                cmd_line = self._create_command_line(
                    multi_card,
                    deepspeed,
                    example_script,
                    model_name,
                    gaudi_config_name,
                    tmp_dir,
                    task=self.TASK_NAME,
                    lr=baseline.get("distribution").get(distribution).get("learning_rate"),
                    train_batch_size=baseline.get("distribution").get(distribution).get("train_batch_size"),
                    eval_batch_size=baseline.get("eval_batch_size"),
                    num_epochs=baseline.get("num_train_epochs"),
                    extra_command_line_arguments=baseline.get("distribution")
                    .get(distribution)
                    .get("extra_arguments", []),
                )

                p = subprocess.Popen(cmd_line)
                return_code = p.wait()

                # Ensure the run finished without any issue
                self.assertEqual(return_code, 0)

                with open(Path(tmp_dir) / "all_results.json") as fp:
                    results = json.load(fp)

                # Ensure performance requirements (accuracy, training time) are met
                self.assert_no_regression(results, baseline.get("distribution").get(distribution))

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

    def _create_command_line(
        self,
        multi_card: bool,
        deepspeed: bool,
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
        task_option = f"--{self.DATASET_PARAMETER_NAME} {task}" if task else " "

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

        cmd_line += [
            f"{script}",
            f"--model_name_or_path {model_name}",
            f"--gaudi_config_name {gaudi_config_name}",
            f"{task_option}",
            "--do_train",
            "--do_eval",
            f"--output_dir {output_dir}",
            "--overwrite_output_dir",
            f"--learning_rate {lr}",
            f"--per_device_train_batch_size {train_batch_size}",
            f"--per_device_eval_batch_size {eval_batch_size}",
            f" --num_train_epochs {num_epochs}",
            "--use_habana",
            "--use_lazy_mode",
            "--throughput_warmup_steps 3",
            "--save_strategy no",
        ]

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

    def assert_no_regression(self, results: Dict, baseline: Dict):
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

        # There is no accuracy metric for `run_clip.py`
        min_number_metrics = 2 if self.EXAMPLE_NAME == "run_clip" else 3

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


class QuestionAnsweringExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_qa"):
    TASK_NAME = "squad"


class MultiCardQuestionAnsweringExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_qa", multi_card=True
):
    TASK_NAME = "squad"


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


# class SpeechRecognitionExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_speech_recognition_ctc"):
#     TASK_NAME = "librispeech_asr"


class MultiCardSpeechRecognitionExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_speech_recognition_ctc", multi_card=True
):
    TASK_NAME = "regisss/librispeech_asr_for_optimum_habana_ci"


class MultiCardSummarizationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_summarization", multi_card=True
):
    TASK_NAME = "cnn_dailymail"


class MultiCardSeq2SeqQuestionAnsweringExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_seq2seq_qa", multi_card=True
):
    TASK_NAME = "squad_v2"


class MultiCardVisionLanguageExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_clip", multi_card=True
):
    TASK_NAME = "ydshieh/coco_dataset_script"


class ProteinFoldingExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_esmfold"):
    pass


class MultiCardCausalLanguageModelingLORAExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_lora_clm"
):
    pass


class TextGenerationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_generation"):
    TASK_NAME = "prompt"
