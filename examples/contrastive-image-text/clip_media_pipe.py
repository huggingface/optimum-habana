# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

import numpy as np
from habana_frameworks.mediapipe import fn
from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.backend.operator_specs import schema
from habana_frameworks.mediapipe.media_types import dtype, ftype, imgtype, randomCropType, readerOutType
from habana_frameworks.mediapipe.mediapipe import MediaPipe
from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode
from habana_frameworks.mediapipe.operators.reader_nodes.read_image_from_dir import get_max_file
from habana_frameworks.torch.hpu import get_device_name
from torch.utils.data.sampler import BatchSampler


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
    Class defining clip media pipe.

    """

    batch_sampler = None
    instance_count = 0

    def __init__(self, is_training=True, dataset=None, sampler=None, batch_size=512, drop_last=False, queue_depth=1):
        self.device = get_device_name()
        self.is_training = is_training
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

        ClipMediaPipe.instance_count += 1

    def definegraph(self):
        res_pp_filter = ftype.BICUBIC
        self.input = fn.ClipDataReader(label_dtype=dtype.UINT32, dataset=self.dataset)
        jpegs, input_ids, attention_masks = self.input()
        def_output_image_size = [self.image_size, self.image_size]
        self.decode = fn.ImageDecoder(
            device=self.device,
            output_format=imgtype.RGB_P,
            random_crop_type=randomCropType.CENTER_CROP,
            resize=def_output_image_size,
            resampling_mode=res_pp_filter,
        )
        images = self.decode(jpegs)
        cmn_pos_offset = 0.5
        normalize_mean = np.array(self.dataset.image_mean).astype(np.float32) * 255
        normalize_std = 1 / (np.array(self.dataset.image_std).astype(np.float32) * 255)
        norm_mean = fn.MediaConst(data=normalize_mean, shape=[1, 1, 3], dtype=dtype.FLOAT32)
        norm_std = fn.MediaConst(data=normalize_std, shape=[1, 1, 3], dtype=dtype.FLOAT32)
        cmn = fn.CropMirrorNorm(
            crop_w=self.image_size,
            crop_h=self.image_size,
            crop_pos_x=cmn_pos_offset,
            crop_pos_y=cmn_pos_offset,
            crop_d=0,
            dtype=dtype.FLOAT32,
        )
        mean = norm_mean()
        std = norm_std()
        images = cmn(images, mean, std)
        return images, input_ids, attention_masks
