# coding=utf-8
# Copyright 2024 The HuggingFace Team All rights reserved.
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


import os

import numpy as np
import torch
from datasets import Dataset as DatasetHF
from torch.distributed import get_rank, get_world_size
from torch.utils.data.sampler import BatchSampler
from transformers.trainer_pt_utils import DistributedSamplerWithLoop

from optimum.utils import logging


logger = logging.get_logger(__name__)


try:
    from habana_frameworks.mediapipe import fn
    from habana_frameworks.mediapipe.media_types import (
        dtype,
        ftype,
        imgtype,
        readerOutType,
    )
    from habana_frameworks.mediapipe.mediapipe import MediaPipe
    from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
    from habana_frameworks.mediapipe.operators.reader_nodes.reader_nodes import (
        media_ext_reader_op_impl,
        media_ext_reader_op_tensor_info,
    )
except ImportError:
    pass


def get_dataset_for_pipeline(img_dir):
    labels = open(f"{img_dir}/label.txt").readlines()
    dct = {"image": [], "text": []}
    for item in sorted(
        [i for i in os.listdir(img_dir) if "txt" not in i],
        key=lambda x: int(x.split(".")[0]),
    ):
        key = int(item.split(".")[0])
        dct["image"] += [f"{img_dir}/{item}"]
        dct["text"] += [labels[key]]

    def gen():
        for idx in range(len(dct["image"])):
            yield {"image": dct["image"][idx], "text": dct["text"][idx]}

    return DatasetHF.from_generator(gen)


class ReadImageTextFromDataset(media_ext_reader_op_impl):
    """
    Class defining read image/text from directory node.
    """

    def __init__(self, params, fw_params):
        priv_params = params["priv_params"]
        self.dataset = priv_params["dataset"]

        self.dataset_image = []
        self.dataset_prompt_embeds = []
        self.dataset_pooled_prompt_embeds = []
        self.dataset_original_sizes = []
        self.dataset_crop_top_lefts = []
        for k in self.dataset:
            self.dataset_image += [k["image"]]
            self.dataset_prompt_embeds += [k["prompt_embeds"]]
            self.dataset_pooled_prompt_embeds += [k["pooled_prompt_embeds"]]
            self.dataset_original_sizes += [k["original_sizes"]]
            self.dataset_crop_top_lefts += [k["crop_top_lefts"]]

        self.dataset_image = np.array(self.dataset_image)
        self.dataset_prompt_embeds = np.array(self.dataset_prompt_embeds, dtype=np.float32)
        self.dataset_pooled_prompt_embeds = np.array(self.dataset_pooled_prompt_embeds, dtype=np.float32)
        self.dataset_original_sizes = np.array(self.dataset_original_sizes, dtype=np.uint32)
        self.dataset_crop_top_lefts = np.array(self.dataset_crop_top_lefts, dtype=np.uint32)
        self.epoch = 0
        self.batch_sampler = priv_params["batch_sampler"]

        self.num_imgs_slice = len(self.batch_sampler.sampler)
        self.num_batches_slice = len(self.batch_sampler)

        logger.info("Finding largest file ...")
        self.max_file = max(self.dataset["image"], key=lambda x: len(x))
        self.batch_size = fw_params.batch_size

    def gen_output_info(self):
        out_info = []
        o = media_ext_reader_op_tensor_info(dtype.NDT, np.array([self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        sample = self.dataset[0]
        sample["pooled_prompt_embeds"]
        d0 = len(sample["pooled_prompt_embeds"])
        d1 = len(sample["prompt_embeds"])
        d2 = len(sample["prompt_embeds"][0])
        o = media_ext_reader_op_tensor_info(dtype.FLOAT32, np.array([d2, d1, self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        o = media_ext_reader_op_tensor_info(dtype.FLOAT32, np.array([d0, self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        o = media_ext_reader_op_tensor_info("uint32", np.array([2, self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        o = media_ext_reader_op_tensor_info("uint32", np.array([2, self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        return out_info

    def get_largest_file(self):
        return self.max_file

    def get_media_output_type(self):
        return readerOutType.FILE_LIST

    def __len__(self):
        return self.num_batches_slice

    def __iter__(self):
        self.iter_loc = 0
        self.epoch += 1
        self.batch_sampler.sampler.set_epoch(
            self.epoch
        )  # Without this dist sampler will create same batches every epoch
        self.batch_sampler_iter = iter(self.batch_sampler)
        return self

    def __next__(self):
        if self.iter_loc > (self.num_imgs_slice - 1):
            raise StopIteration

        data_idx = next(self.batch_sampler_iter)
        img_list = list(self.dataset_image[data_idx])
        prompt_embeds_np = self.dataset_prompt_embeds[data_idx]
        pooled_prompt_embeds_np = self.dataset_pooled_prompt_embeds[data_idx]
        original_sizes = self.dataset_original_sizes[data_idx]
        crop_top_lefts = self.dataset_crop_top_lefts[data_idx]

        self.iter_loc = self.iter_loc + self.batch_size
        return (
            img_list,
            prompt_embeds_np,
            pooled_prompt_embeds_np,
            original_sizes,
            crop_top_lefts,
        )


class RandomFlipFunction(media_function):
    """
    Class to randomly generate input for RandomFlip media node.

    """

    def __init__(self, params):
        """
        :params params: random_flip_func specific params.
                        shape: output shape
                        dtype: output data type
                        seed: seed to be used
        """
        self.np_shape = params["shape"][::-1]
        self.np_dtype = params["dtype"]
        self.seed = params["seed"]
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        :returns : randomly generated binary output per image.
        """
        probabilities = [1.0 - 0.5, 0.5]
        random_flips = self.rng.choice([0, 1], p=probabilities, size=self.np_shape)
        random_flips = np.array(random_flips, dtype=self.np_dtype)
        return random_flips


class SDXLMediaPipe(MediaPipe):
    """
    Class defining SDXL media pipe:
        read data --> image decoding (include crop and resize) --> crop mirror normalize

    Original set of PyTorch transformations:
        aspect ratio preserving resize -> center crop -> normalize
    """

    instance_count = 0

    def __init__(
        self,
        dataset=None,
        image_size=512,
        sampler=None,
        batch_size=512,
        drop_last=True,
        queue_depth=5,
    ):
        self.device = "legacy"
        self.dataset = dataset
        self.batch_size = batch_size

        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)

        self.image_size = image_size

        pipe_name = "{}:{}".format(self.__class__.__name__, SDXLMediaPipe.instance_count)
        pipe_name = str(pipe_name)

        super(SDXLMediaPipe, self).__init__(
            device=self.device,
            batch_size=batch_size,
            prefetch_depth=queue_depth,
            pipe_name=pipe_name,
        )

        priv_params = {}
        priv_params["dataset"] = self.dataset
        priv_params["batch_sampler"] = self.batch_sampler

        self.input = fn.MediaExtReaderOp(impl=ReadImageTextFromDataset, num_outputs=5, priv_params=priv_params)

        def_output_image_size = [self.image_size, self.image_size]
        res_pp_filter = ftype.BI_LINEAR
        self.decode = fn.ImageDecoder(
            device="hpu",
            output_format=imgtype.RGB_P,
            # random_crop_type=randomCropType.CENTER_CROP,
            resize=def_output_image_size,
            resampling_mode=res_pp_filter,
        )
        normalize_mean = np.array([255 / 2, 255 / 2, 255 / 2]).astype(np.float32)
        normalize_std = 1 / (np.array([255 / 2, 255 / 2, 255 / 2]).astype(np.float32))
        norm_mean = fn.MediaConst(data=normalize_mean, shape=[1, 1, 3], dtype=dtype.FLOAT32)
        norm_std = fn.MediaConst(data=normalize_std, shape=[1, 1, 3], dtype=dtype.FLOAT32)
        self.cmn = fn.CropMirrorNorm(
            crop_w=self.image_size,
            crop_h=self.image_size,
            crop_pos_x=0,
            crop_pos_y=0,
            crop_d=0,
            dtype=dtype.FLOAT32,
        )
        self.mean = norm_mean()
        self.std = norm_std()

        self.random_flip_input = fn.MediaFunc(
            func=RandomFlipFunction,
            shape=[self.batch_size],
            dtype=dtype.UINT8,
            seed=100,
        )
        self.random_flip = fn.RandomFlip(horizontal=1)

        SDXLMediaPipe.instance_count += 1

    def definegraph(self):
        jpegs, prompt_embeds, pooled_prompt_embeds, original_sizes, crop_top_lefts = self.input()
        images = self.decode(jpegs)
        flip = self.random_flip_input()
        images = self.random_flip(images, flip)
        images = self.cmn(images, self.mean, self.std)
        return (
            images,
            prompt_embeds,
            pooled_prompt_embeds,
            original_sizes,
            crop_top_lefts,
        )


class MediaApiDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        resolution,
        batch_size=1,
    ):
        self.dataset = dataset

        from habana_frameworks.mediapipe.plugins.iterator_pytorch import (
            HPUGenericPytorchIterator,
        )

        try:
            world_size = get_world_size()
        except Exception:
            world_size = 1

        if world_size > 1:
            process_index = get_rank()
            self.sampler = DistributedSamplerWithLoop(
                self.dataset,
                num_replicas=world_size,
                rank=process_index,
                seed=1,
                batch_size=batch_size,
            )
        else:
            self.sampler = torch.utils.data.sampler.RandomSampler(self.dataset)

        pipeline = SDXLMediaPipe(
            dataset=dataset,
            image_size=resolution,
            sampler=self.sampler,
            batch_size=batch_size,
            drop_last=True,
            queue_depth=5,
        )
        self.iterator = HPUGenericPytorchIterator(mediapipe=pipeline)
        self.epoch = 0

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        self.iterator.__iter__()
        self.epoch += 1
        return self

    def __next__(self):
        data = next(self.iterator)
        return {
            "pixel_values": data[0],
            "prompt_embeds": data[1],
            "pooled_prompt_embeds": data[2],
            "original_sizes": data[3],
            "crop_top_lefts": data[4],
        }
