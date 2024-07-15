# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

"""
Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/auto_pipeline.py
- Added GAUDI_PREFIX_NAME to support Gaudi pipeline in _gaudi_get_task_class.
- Only AutoPipelineForText2Image and AutoPipelineForInpainting are retained, and reimplement the from_pretrained and from_pipe to support the Gaudi pipelines.
"""

from collections import OrderedDict

from diffusers.pipelines import (
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    auto_pipeline,
)
from huggingface_hub.utils import validate_hf_hub_args

from .controlnet.pipeline_controlnet import GaudiStableDiffusionControlNetPipeline
from .stable_diffusion.pipeline_stable_diffusion import GaudiStableDiffusionPipeline
from .stable_diffusion.pipeline_stable_diffusion_inpaint import GaudiStableDiffusionInpaintPipeline
from .stable_diffusion_xl.pipeline_stable_diffusion_xl import GaudiStableDiffusionXLPipeline
from .stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import GaudiStableDiffusionXLInpaintPipeline


GAUDI_PREFIX_NAME = "Gaudi"

GAUDI_AUTO_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", GaudiStableDiffusionPipeline),
        ("stable-diffusion-xl", GaudiStableDiffusionXLPipeline),
        ("stable-diffusion-controlnet", GaudiStableDiffusionControlNetPipeline),
    ]
)


GAUDI_AUTO_INPAINT_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", GaudiStableDiffusionInpaintPipeline),
        ("stable-diffusion-xl", GaudiStableDiffusionXLInpaintPipeline),
    ]
)


GAUDI_SUPPORTED_TASKS_MAPPINGS = [
    GAUDI_AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    GAUDI_AUTO_INPAINT_PIPELINES_MAPPING,
]


def _gaudi_get_task_class(mapping, pipeline_class_name, throw_error_if_not_exist: bool = True):
    def get_model(pipeline_class_name):
        for task_mapping in GAUDI_SUPPORTED_TASKS_MAPPINGS:
            for model_name, pipeline in task_mapping.items():
                if pipeline.__name__ == pipeline_class_name:
                    return model_name

    pipeline_class_name = GAUDI_PREFIX_NAME + pipeline_class_name
    model_name = get_model(pipeline_class_name)

    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class

    if throw_error_if_not_exist:
        raise ValueError(f"AutoPipeline can't find a pipeline linked to {pipeline_class_name} for {model_name}")


class AutoPipelineForText2Image(AutoPipelineForText2Image):
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        orig_supported_mappings = auto_pipeline.SUPPORTED_TASKS_MAPPINGS
        orig_txt2img_mappings = auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING
        orig_func = auto_pipeline._get_task_class
        auto_pipeline.SUPPORTED_TASKS_MAPPINGS = GAUDI_SUPPORTED_TASKS_MAPPINGS
        auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING = GAUDI_AUTO_TEXT2IMAGE_PIPELINES_MAPPING
        auto_pipeline._get_task_class = _gaudi_get_task_class
        pipeline = super().from_pretrained(pretrained_model_or_path, **kwargs)
        auto_pipeline.SUPPORTED_TASKS_MAPPINGS = orig_supported_mappings
        auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING = orig_txt2img_mappings
        auto_pipeline._get_task_class = orig_func
        return pipeline

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        orig_supported_mappings = auto_pipeline.SUPPORTED_TASKS_MAPPINGS
        orig_txt2img_mappings = auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING
        orig_func = auto_pipeline._get_task_class
        auto_pipeline.SUPPORTED_TASKS_MAPPINGS = GAUDI_SUPPORTED_TASKS_MAPPINGS
        auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING = GAUDI_AUTO_TEXT2IMAGE_PIPELINES_MAPPING
        auto_pipeline._get_task_class = _gaudi_get_task_class
        model = super().from_pipe(pipeline, **kwargs)
        auto_pipeline.SUPPORTED_TASKS_MAPPINGS = orig_supported_mappings
        auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING = orig_txt2img_mappings
        auto_pipeline._get_task_class = orig_func
        return model


class AutoPipelineForInpainting(AutoPipelineForInpainting):
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        orig_supported_mappings = auto_pipeline.SUPPORTED_TASKS_MAPPINGS
        orig_inpaint_mappings = auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING
        orig_func = auto_pipeline._get_task_class
        auto_pipeline.SUPPORTED_TASKS_MAPPINGS = GAUDI_SUPPORTED_TASKS_MAPPINGS
        auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING = GAUDI_AUTO_INPAINT_PIPELINES_MAPPING
        auto_pipeline._get_task_class = _gaudi_get_task_class
        pipeline = super().from_pretrained(pretrained_model_or_path, **kwargs)
        auto_pipeline.SUPPORTED_TASKS_MAPPINGS = orig_supported_mappings
        auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING = orig_inpaint_mappings
        auto_pipeline._get_task_class = orig_func
        return pipeline

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        orig_supported_mappings = auto_pipeline.SUPPORTED_TASKS_MAPPINGS
        orig_inpaint_mappings = auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING
        orig_func = auto_pipeline._get_task_class
        auto_pipeline.SUPPORTED_TASKS_MAPPINGS = GAUDI_SUPPORTED_TASKS_MAPPINGS
        auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING = GAUDI_AUTO_INPAINT_PIPELINES_MAPPING
        auto_pipeline._get_task_class = _gaudi_get_task_class
        model = super().from_pipe(pipeline, **kwargs)
        auto_pipeline.SUPPORTED_TASKS_MAPPINGS = orig_supported_mappings
        auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING = orig_inpaint_mappings
        auto_pipeline._get_task_class = orig_func
        return model
