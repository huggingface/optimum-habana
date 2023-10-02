# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import importlib
import inspect
import os
import sys
import tempfile
import warnings
from typing import Optional, Union

import torch
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo

from optimum.utils import logging

from ...transformers.gaudi_configuration import GaudiConfig


logger = logging.get_logger(__name__)


GAUDI_LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
    "optimum.habana.diffusers.schedulers": {
        "GaudiDDIMScheduler": ["save_pretrained", "from_pretrained"],
    },
}

GAUDI_ALL_IMPORTABLE_CLASSES = {}
for library in GAUDI_LOADABLE_CLASSES:
    GAUDI_ALL_IMPORTABLE_CLASSES.update(GAUDI_LOADABLE_CLASSES[library])


class GaudiDiffusionPipeline(DiffusionPipeline):
    """
    Extends the [`DiffusionPipeline`](https://huggingface.co/docs/diffusers/api/diffusion_pipeline) class:
    - The pipeline is initialized on Gaudi if `use_habana=True`.
    - The pipeline's Gaudi configuration is saved and pushed to the hub.

    Args:
        use_habana (bool, defaults to `False`):
            Whether to use Gaudi (`True`) or CPU (`False`).
        use_hpu_graphs (bool, defaults to `False`):
            Whether to use HPU graphs or not.
        gaudi_config (Union[str, [`GaudiConfig`]], defaults to `None`):
            Gaudi configuration to use. Can be a string to download it from the Hub.
            Or a previously initialized config can be passed.
        bf16_full_eval (bool, defaults to `False`):
            Whether to use full bfloat16 evaluation instead of 32-bit.
            This will be faster and save memory compared to fp32/mixed precision but can harm generated images.
    """

    def __init__(
        self,
        use_habana: bool = False,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
        bf16_full_eval: bool = False,
    ):
        super().__init__()

        self.use_habana = use_habana
        if self.use_habana:
            self.use_hpu_graphs = use_hpu_graphs
            if self.use_hpu_graphs:
                logger.info("Enabled HPU graphs.")
            else:
                logger.info("Enabled lazy mode because `use_hpu_graphs=False`.")

            self._device = torch.device("hpu")

            if isinstance(gaudi_config, str):
                # Config from the Hub
                self.gaudi_config = GaudiConfig.from_pretrained(gaudi_config)
            elif isinstance(gaudi_config, GaudiConfig):
                # Config already initialized
                self.gaudi_config = copy.deepcopy(gaudi_config)
            else:
                raise ValueError(
                    f"`gaudi_config` must be a string or a GaudiConfig object but is {type(gaudi_config)}."
                )

            if self.gaudi_config.use_habana_mixed_precision or self.gaudi_config.use_torch_autocast:
                if bf16_full_eval:
                    logger.warning(
                        "`use_habana_mixed_precision` or `use_torch_autocast` is True in the given Gaudi configuration but "
                        "`torch_dtype=torch.blfloat16` was given. Disabling mixed precision and continuing in bf16 only."
                    )
                    self.gaudi_config.use_torch_autocast = False
                    self.gaudi_config.use_habana_mixed_precision = False
                elif self.gaudi_config.use_torch_autocast:
                    # Open temporary files to write mixed-precision ops
                    with tempfile.NamedTemporaryFile() as hmp_bf16_file:
                        with tempfile.NamedTemporaryFile() as hmp_fp32_file:
                            self.gaudi_config.write_bf16_fp32_ops_to_text_files(
                                hmp_bf16_file.name,
                                hmp_fp32_file.name,
                            )
                            os.environ["LOWER_LIST"] = str(hmp_bf16_file)
                            os.environ["FP32_LIST"] = str(hmp_fp32_file)

                            import habana_frameworks.torch.core  # noqa
                elif self.gaudi_config.use_habana_mixed_precision:
                    try:
                        from habana_frameworks.torch.hpex import hmp
                    except ImportError as error:
                        error.msg = f"Could not import habana_frameworks.torch.hpex. {error.msg}."
                        raise error

                    warnings.warn(
                        "Habana Mixed Precision is deprecated and will be removed in SynapseAI v1.12. Please"
                        " use Torch Autocast instead setting `use_torch_autocast=true` in your Gaudi configuration.",
                        FutureWarning,
                    )

                    # Open temporary files to write mixed-precision ops
                    with tempfile.NamedTemporaryFile() as hmp_bf16_file:
                        with tempfile.NamedTemporaryFile() as hmp_fp32_file:
                            # hmp.convert needs ops to be written in text files
                            self.gaudi_config.write_bf16_fp32_ops_to_text_files(
                                hmp_bf16_file.name,
                                hmp_fp32_file.name,
                            )
                            hmp.convert(
                                opt_level=self.gaudi_config.hmp_opt_level,
                                bf16_file_path=hmp_bf16_file.name,
                                fp32_file_path=hmp_fp32_file.name,
                                isVerbose=self.gaudi_config.hmp_is_verbose,
                            )

            # Workaround for Synapse 1.11 for full bf16 and Torch Autocast
            if bf16_full_eval or self.gaudi_config.use_torch_autocast:
                import diffusers

                from ..models import gaudi_unet_2d_condition_model_forward

                diffusers.models.unet_2d_condition.UNet2DConditionModel.forward = gaudi_unet_2d_condition_model_forward

            if self.use_hpu_graphs:
                try:
                    import habana_frameworks.torch as ht
                except ImportError as error:
                    error.msg = f"Could not import habana_frameworks.torch. {error.msg}."
                    raise error
                self.ht = ht
                self.hpu_stream = ht.hpu.Stream()
                self.cache = {}
            else:
                try:
                    import habana_frameworks.torch.core as htcore
                except ImportError as error:
                    error.msg = f"Could not import habana_frameworks.torch.core. {error.msg}."
                    raise error
                self.htcore = htcore
        else:
            if use_hpu_graphs:
                raise ValueError(
                    "`use_hpu_graphs` is True but `use_habana` is False, please set `use_habana=True` to use HPU"
                    " graphs."
                )
            if gaudi_config is not None:
                raise ValueError(
                    "Got a non-None `gaudi_config` but `use_habana` is False, please set `use_habana=True` to use this"
                    " Gaudi configuration."
                )
            logger.info("Running on CPU.")
            self._device = torch.device("cpu")

    def register_modules(self, **kwargs):
        # import it here to avoid circular import
        from diffusers import pipelines

        for name, module in kwargs.items():
            # retrieve library
            if module is None:
                register_dict = {name: (None, None)}
            else:
                # register the config from the original module, not the dynamo compiled one
                if is_compiled_module(module):
                    not_compiled_module = module._orig_mod
                else:
                    not_compiled_module = module

                library = not_compiled_module.__module__.split(".")[0]
                if library == "optimum":
                    library = "optimum.habana.diffusers.schedulers"

                # check if the module is a pipeline module
                module_path_items = not_compiled_module.__module__.split(".")
                pipeline_dir = module_path_items[-2] if len(module_path_items) > 2 else None

                path = not_compiled_module.__module__.split(".")
                is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

                # if library is not in GAUDI_LOADABLE_CLASSES, then it is a custom module.
                # Or if it's a pipeline module, then the module is inside the pipeline
                # folder so we set the library to module name.
                if is_pipeline_module:
                    library = pipeline_dir
                elif library not in GAUDI_LOADABLE_CLASSES:
                    library = not_compiled_module.__module__

                # retrieve class_name
                class_name = not_compiled_module.__class__.__name__

                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save the pipeline and Gaudi configurations.
        More information [here](https://huggingface.co/docs/diffusers/api/diffusion_pipeline#diffusers.DiffusionPipeline.save_pretrained).

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name", None)
        model_index_dict.pop("_diffusers_version", None)
        model_index_dict.pop("_module", None)
        model_index_dict.pop("_name_or_path", None)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", False)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}

        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            # Dynamo wraps the original model in a private class.
            # I didn't find a public API to get the original class.
            if is_compiled_module(sub_model):
                sub_model = sub_model._orig_mod
                model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in GAUDI_LOADABLE_CLASSES
            for library_name, library_classes in GAUDI_LOADABLE_CLASSES.items():
                if library_name in sys.modules:
                    library = importlib.import_module(library_name)
                else:
                    logger.info(
                        f"{library_name} is not installed. Cannot save {pipeline_component_name} as {library_classes} from {library_name}"
                    )

                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in GAUDI_LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            if save_method_name is None:
                logger.warn(f"self.{pipeline_component_name}={sub_model} of type {type(sub_model)} cannot be saved.")
                # make sure that unsaveable components are not tried to be loaded afterward
                self.register_to_config(**{pipeline_component_name: (None, None)})
                continue

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)

        # finally save the config
        self.save_config(save_directory)
        if hasattr(self, "gaudi_config"):
            self.gaudi_config.save_pretrained(save_directory)

        if push_to_hub:
            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        """
        More information [here](https://huggingface.co/docs/diffusers/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained).
        """

        # Set the correct log level depending on the node
        # Already done in super().init() but we have to do it again
        # because we use optimum.utils.logging here and not
        # diffusers.utils.logging
        log_level = kwargs.pop("log_level", logging.INFO)
        logging.set_verbosity(log_level)
        logging.enable_default_handler()
        logging.enable_explicit_format()

        # Import diffusers.pipelines.pipeline_utils to override the values of LOADABLE_CLASSES and ALL_IMPORTABLE_CLASSES
        import diffusers.pipelines.pipeline_utils

        diffusers.pipelines.pipeline_utils.LOADABLE_CLASSES = GAUDI_LOADABLE_CLASSES
        diffusers.pipelines.pipeline_utils.ALL_IMPORTABLE_CLASSES = GAUDI_ALL_IMPORTABLE_CLASSES

        # Define a new kwarg here to know in the __init__ whether to use full bf16 precision or not
        bf16_full_eval = kwargs.get("torch_dtype", None) == torch.bfloat16
        kwargs["bf16_full_eval"] = bf16_full_eval

        return super().from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
