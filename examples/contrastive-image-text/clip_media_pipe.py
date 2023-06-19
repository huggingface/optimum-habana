# coding=utf-8
# Copyright 2023 The HuggingFace Team All rights reserved.
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

import numpy as np
from torch.utils.data.sampler import BatchSampler


try:
    from habana_frameworks.mediapipe import fn
    from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
    from habana_frameworks.mediapipe.backend.operator_specs import schema
    from habana_frameworks.mediapipe.media_types import dtype, ftype, imgtype, randomCropType, readerOutType
    from habana_frameworks.mediapipe.mediapipe import MediaPipe
    from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode
    from habana_frameworks.mediapipe.operators.reader_nodes.read_image_from_dir import get_max_file
    from habana_frameworks.torch.hpu import get_device_name
except ImportError:
    pass


class read_image_text_from_dataset(MediaReaderNode):
    """
    Class defining read image/text from directory node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr):
        super().__init__(name, guid, device, inputs, params, cparams, node_attr)
        self.meta_dtype = params["label_dtype"]
        self.dataset = params["dataset"]
        self.epoch = 0

        self.num_imgs_slice = len(ClipMediaPipe.batch_sampler.sampler)
        self.num_batches_slice = len(ClipMediaPipe.batch_sampler)
        print("Finding largest file ...")
        self.max_file = get_max_file(self.dataset["image_path"])
        print("largest file is ", self.max_file)

    def set_params(self, params):
        self.batch_size = params.batch_size

    def gen_output_info(self):
        out_info = []
        o = opnode_tensor_info(dtype.NDT, np.array([self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        o = opnode_tensor_info(
            self.meta_dtype, np.array([self.dataset.text_max_length, self.batch_size], dtype=np.uint32), ""
        )
        out_info.append(o)
        o = opnode_tensor_info(
            self.meta_dtype, np.array([self.dataset.text_max_length, self.batch_size], dtype=np.uint32), ""
        )
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
        self.batch_sampler_iter = iter(ClipMediaPipe.batch_sampler)
        self.epoch += 1
        return self

    def __next__(self):
        if self.iter_loc > (self.num_imgs_slice - 1):
            raise StopIteration

        data_idx = next(self.batch_sampler_iter)
        data = self.dataset.__getitems__(data_idx)
        img_list = []

        input_id_list = np.zeros(shape=(self.batch_size, self.dataset.text_max_length), dtype=self.meta_dtype)
        attention_mask_list = np.zeros(shape=(self.batch_size, self.dataset.text_max_length), dtype=self.meta_dtype)
        for i, x in enumerate(data):
            img_list.append(x["image_path"])
            input_id_list[i, :] = x["input_ids"]
            attention_mask_list[i, :] = x["attention_mask"]

        self.iter_loc = self.iter_loc + self.batch_size

        return img_list, input_id_list, attention_mask_list


read_image_text_from_dataset_params = {
    "label_dtype": dtype.UINT64,
    "dataset": None,
}
schema.add_operator(
    "ClipDataReader",
    None,
    0,
    0,
    [],
    3,
    read_image_text_from_dataset_params,
    None,
    read_image_text_from_dataset,
    dtype.NDT,
)
op_class = fn.operator_add("ClipDataReader")
op_class.__module__ = fn.__name__
setattr(fn, "ClipDataReader", op_class)


class ClipMediaPipe(MediaPipe):
    """
    Class defining clip media pipe:
        read data --> image decoding (include crop and resize) --> crop mirror normalize

    Original set of PyTorch transformations:
        aspect ratio preserving resize -> center crop -> normalize

    """

    batch_sampler = None
    instance_count = 0

    def __init__(self, dataset=None, sampler=None, batch_size=512, drop_last=False, queue_depth=1):
        self.device = get_device_name()
        self.dataset = dataset
        self.drop_last = drop_last
        self.sampler = sampler
        ClipMediaPipe.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.image_size = self.dataset.image_resize

        pipe_name = "{}:{}".format(self.__class__.__name__, ClipMediaPipe.instance_count)
        pipe_name = str(pipe_name)

        super(ClipMediaPipe, self).__init__(
            device=self.device, batch_size=batch_size, prefetch_depth=queue_depth, pipe_name=pipe_name
        )

        self.input = fn.ClipDataReader(label_dtype=dtype.UINT32, dataset=self.dataset)
        def_output_image_size = [self.image_size, self.image_size]
        res_pp_filter = ftype.BICUBIC
        self.decode = fn.ImageDecoder(
            device=self.device,
            output_format=imgtype.RGB_P,
            random_crop_type=randomCropType.CENTER_CROP,
            resize=def_output_image_size,
            resampling_mode=res_pp_filter,
        )

        cmn_pos_offset = 0.5
        normalize_mean = np.array(self.dataset.image_mean).astype(np.float32) * 255
        normalize_std = 1 / (np.array(self.dataset.image_std).astype(np.float32) * 255)
        norm_mean = fn.MediaConst(data=normalize_mean, shape=[1, 1, 3], dtype=dtype.FLOAT32)
        norm_std = fn.MediaConst(data=normalize_std, shape=[1, 1, 3], dtype=dtype.FLOAT32)
        self.cmn = fn.CropMirrorNorm(
            crop_w=self.image_size,
            crop_h=self.image_size,
            crop_pos_x=cmn_pos_offset,
            crop_pos_y=cmn_pos_offset,
            crop_d=0,
            dtype=dtype.FLOAT32,
        )
        self.mean = norm_mean()
        self.std = norm_std()

        ClipMediaPipe.instance_count += 1

    def definegraph(self):
        jpegs, input_ids, attention_masks = self.input()
        images = self.decode(jpegs)
        images = self.cmn(images, self.mean, self.std)
        return images, input_ids, attention_masks
