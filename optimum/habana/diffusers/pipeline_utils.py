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
import os
import tempfile
from typing import Optional, Union

import torch

from diffusers import DiffusionPipeline
from optimum.utils import logging

from ..transformers.gaudi_configuration import GAUDI_CONFIG_NAME, GaudiConfig


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
    """

    def __init__(
        self,
        use_habana: bool = False,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
    ):
        super().__init__()

        self.use_habana = use_habana
        # HPU graphs and Gaudi configuration require `use_habana` to be True
        if not use_habana:
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
        self.use_hpu_graphs = use_hpu_graphs
        self.gaudi_config = gaudi_config

        if self.use_habana:
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

            if self.use_hpu_graphs:
                try:
                    import habana_frameworks.torch as ht
                except ImportError as error:
                    error.msg = f"Could not import habana_frameworks.torch. {error.msg}."
                    raise error
                self.ht = ht
                self.hpu_graph = ht.hpu.HPUGraph()
                self.hpu_stream = ht.hpu.Stream()
                self.static_inputs = list()
                self.static_outputs = None
            else:
                try:
                    import habana_frameworks.torch.core as htcore
                except ImportError as error:
                    error.msg = f"Could not import habana_frameworks.torch.core. {error.msg}."
                    raise error
                self.htcore = htcore

            if self.gaudi_config.use_habana_mixed_precision:
                try:
                    from habana_frameworks.torch.hpex import hmp
                except ImportError as error:
                    error.msg = f"Could not import habana_frameworks.torch.hpex. {error.msg}."
                    raise error
                self.hmp = hmp

                # Open temporary files to mixed-precision write ops
                with tempfile.NamedTemporaryFile() as hmp_bf16_file:
                    with tempfile.NamedTemporaryFile() as hmp_fp32_file:
                        # hmp.convert needs ops to be written in text files
                        self.gaudi_config.write_bf16_fp32_ops_to_text_files(
                            hmp_bf16_file.name,
                            hmp_fp32_file.name,
                        )
                        self.hmp.convert(
                            opt_level=self.gaudi_config.hmp_opt_level,
                            bf16_file_path=hmp_bf16_file.name,
                            fp32_file_path=hmp_fp32_file.name,
                            isVerbose=self.gaudi_config.hmp_is_verbose,
                        )
        else:
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
                library = module.__module__.split(".")[0]
                if library == "optimum":
                    library = "optimum.habana.diffusers.schedulers"

                # check if the module is a pipeline module
                pipeline_dir = module.__module__.split(".")[-2] if len(module.__module__.split(".")) > 2 else None
                path = module.__module__.split(".")
                is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

                # if library is not in GAUDI_LOADABLE_CLASSES, then it is a custom module.
                # Or if it's a pipeline module, then the module is inside the pipeline
                # folder so we set the library to module name.
                if library not in GAUDI_LOADABLE_CLASSES or is_pipeline_module:
                    library = pipeline_dir

                # retrieve class_name
                class_name = module.__class__.__name__

                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save the pipeline and Gaudi configurations.
        More information [here](https://huggingface.co/docs/diffusers/api/diffusion_pipeline#diffusers.DiffusionPipeline.save_pretrained).

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        self.save_config(save_directory)
        if hasattr(self, "gaudi_config"):
            self.gaudi_config.save_pretrained(save_directory)

        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name")
        model_index_dict.pop("_diffusers_version")
        model_index_dict.pop("_module", None)

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

            save_method_name = None
            # search for the model's base class in GAUDI_LOADABLE_CLASSES
            for library_name, library_classes in GAUDI_LOADABLE_CLASSES.items():
                library = importlib.import_module(library_name)
                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in GAUDI_LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            save_method = getattr(sub_model, save_method_name)
            save_method(os.path.join(save_directory, pipeline_component_name))

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

        # Import diffusers.pipeline_utils to override the values of LOADABLE_CLASSES and ALL_IMPORTABLE_CLASSES
        import diffusers.pipeline_utils

        diffusers.pipeline_utils.LOADABLE_CLASSES = GAUDI_LOADABLE_CLASSES
        diffusers.pipeline_utils.ALL_IMPORTABLE_CLASSES = GAUDI_ALL_IMPORTABLE_CLASSES

        return super().from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
