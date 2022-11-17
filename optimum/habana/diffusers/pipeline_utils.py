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
from typing import Union

import torch

from diffusers import DiffusionPipeline
from diffusers.pipeline_utils import LOADABLE_CLASSES
from optimum.utils import logging

from ..transformers.gaudi_configuration import GAUDI_CONFIG_NAME, GaudiConfig


logger = logging.get_logger(__name__)


class GaudiDiffusionPipeline(DiffusionPipeline):
    """
    Extends the `DiffusionPipeline` class:
    - The pipeline is initialized on Gaudi.
    - The pipeline's Gaudi configuration is saved and pushed to the hub.

    Args:
        use_habana (bool):
            Whether to use Gaudi (`True`) or CPU (`False`).
        use_lazy_mode (bool):
            Whether to use lazy (`True`) or eager (`False`) mode.
        use_hpu_graphs (bool):
            Whether to use HPU graphs or not.
        gaudi_config (Union[str, [`GaudiConfig`]]):
            Gaudi configuration to use. Can be a string to download it from the Hub.
            Or a previously initialized config can be passed.
        log_level (int):
            The numeric level of the logging event.
    """

    def __init__(
        self,
        use_habana: bool = False,
        use_lazy_mode: bool = False,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
        log_level: int = logging.INFO,
    ):
        super().__init__()

        # Set the correct log level depending on the node
        # Already done in super().init() but we have to do it again
        # because we use optimum.utils.logging here and not
        # diffusers.utils.logging
        logging.set_verbosity(log_level)
        logging.enable_default_handler()
        logging.enable_explicit_format()

        # Lazy mode, HPU graphs and Gaudi configuration should be off if not using HPU
        self.use_habana = use_habana
        self.use_lazy_mode = False if not use_habana else use_lazy_mode
        self.use_hpu_graphs = False if not use_habana else use_hpu_graphs
        self.gaudi_config = None if not use_habana else gaudi_config

        if self.use_lazy_mode and self.use_hpu_graphs:
            raise ValueError("`use_lazy_mode` and `use_hpu_graphs` cannot be both True.")

        if self.use_habana:
            if self.use_lazy_mode:
                logger.info("Enabled lazy mode.")
            elif self.use_hpu_graphs:
                logger.info("Enabled HPU graphs.")
            else:
                os.environ["PT_HPU_LAZY_MODE"] = "2"
                logger.info("Enabled eager mode because use_lazy_mode=False.")

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
            elif self.use_lazy_mode:
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

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~DiffusionPipeline.from_pretrained`]` class method.
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

        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            if sub_model is None:
                # edge case for saving a pipeline with safety_checker=None
                continue

            model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                library = importlib.import_module(library_name)
                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class)
                    if issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            save_method = getattr(sub_model, save_method_name)
            save_method(os.path.join(save_directory, pipeline_component_name))
