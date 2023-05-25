# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

import os
import time

import numpy as np
import torch
import torch.distributed as dist
from habana_frameworks.mediapipe import fn  # NOQA  # NOQA
from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.backend.operator_specs import schema
from habana_frameworks.mediapipe.media_types import decoderStage as ds  # NOQA
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA
from habana_frameworks.mediapipe.media_types import ftype as ft  # NOQA
from habana_frameworks.mediapipe.media_types import imgtype as it  # NOQA
from habana_frameworks.mediapipe.media_types import layout as lt  # NOQA
from habana_frameworks.mediapipe.media_types import randomCropType as rct  # NOQA
from habana_frameworks.mediapipe.media_types import readerOutType as ro
from habana_frameworks.mediapipe.mediapipe import MediaPipe  # NOQA
from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode
from habana_frameworks.mediapipe.operators.reader_nodes.coco_reader import roundup_keylist
from habana_frameworks.mediapipe.operators.reader_nodes.reader_node_params import coco_reader_params
from habana_frameworks.mediapipe.plugins.iterator_pytorch import HPUGenericPytorchIterator


def rounddown_keylist(img_dict, key_list, round_downto):
    """
    Method to round down key list and img dictionary.
    """
    num_key = len(key_list)

    slice_end = int((num_key) / round_downto) * round_downto
    if slice_end == 0:
        raise ValueError("round down failed for img and key list")
    key_list = key_list[0:slice_end]
    num_images = len(key_list)
    img_dict_new = {}
    for index in range(num_images):
        image_id, annotkey = key_list[index]
        img_dict_new[image_id] = img_dict[image_id]

    return img_dict_new, key_list


# Implement a datareader for COCO dataset
class clip_coco_reader(MediaReaderNode):
    """
    Class defining coco reader node.
    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr):
        """
        Constructor method.
        :params name: node name.
        :params guid: guid of node.
        :params device: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(name, guid, device, inputs, params, cparams, node_attr)
        self.prep_caption_data(params["dataset"])
        for attr in [
            "root",
            "slice_index",
            "num_slices",
            "drop_remainder",
            "pad_remainder",
            "seed",
            "shuffle",
            "max_file",
            "partial_batch",
        ]:
            setattr(self, attr, params[attr])
        if self.seed is None:
            # max supported seed value is 32bit so modulo
            self.seed = int(time.time_ns() % (2**31 - 1))

        self.rng = np.random.default_rng(self.seed)
        self.keys = self.shuffler(list(self.id_tokenized_caption_map1.keys()))

        if self.num_slices < 1:
            raise ValueError("num slice cannot be less then 1")
        if self.slice_index >= self.num_slices:
            raise ValueError("slice_index cannot be >= num_slices")
        self.num_imgs = len(self.images)
        if self.num_imgs == 0:
            raise ValueError("image list is empty")

        print("coco_reader seed {} shuffle {}".format(self.seed, self.shuffle))
        print("Total images ", self.num_imgs)
        print("num_slices {} slice_index {}".format(self.num_slices, self.slice_index))
        self.images_slice = self.images
        self.keys_slice = self.keys
        imgids = set(self.images_slice.keys())
        imgids_from_compositekey = {k[0] for k in self.keys}
        assert imgids == imgids_from_compositekey
        self.keys_slice_pad = np.zeros(shape=(0), dtype=int)
        self.round_slice_list(self.num_slices, False)
        # now we slice the dataset
        self.num_keys_slice = int(self.num_keys_slice / self.num_slices)
        idx = np.arange(self.num_keys_slice)
        idx = (idx * self.num_slices) + self.slice_index
        self.keys_slice = [self.keys_slice[i] for i in idx]
        img_list_new = {}
        for index in range(self.num_keys_slice):
            image_id, annotkey = self.keys_slice[index]
            img_list_new[image_id] = self.images_slice[image_id]
        self.images_slice = img_list_new
        print("sliced length {}".format(self.num_keys_slice))
        if self.max_file is None:
            print("Finding largest file ...")
            img_list = (os.path.join(self.root, x) for x in self.images_slice.values())
            self.max_file = max(img_list, key=lambda x: os.stat(x).st_size)
        print("largest file is ", self.max_file)

    def prep_caption_data(self, dataset):
        t0 = time.time()
        self.id_tokenized_caption_map1 = {}
        self.id_tokenized_caption_map2 = {}
        img_caption_count = {}
        self.images = {}
        for k in dataset:
            filename = k["image_path"].split("/")[-1]
            imgkey = int(filename.split(".")[0])
            if imgkey in self.images:
                assert filename == self.images[imgkey]
            else:
                self.images[imgkey] = filename
            count = img_caption_count.get(imgkey, 0)
            key = (imgkey, count)
            assert key not in self.id_tokenized_caption_map1 and key not in self.id_tokenized_caption_map2
            self.id_tokenized_caption_map1[key] = k["input_ids"]
            self.id_tokenized_caption_map2[key] = k["attention_mask"]
            img_caption_count[imgkey] = count + 1
        print("Final time:", time.time() - t0)

    def shuffler(self, arr):
        if self.shuffle:
            print("Shuffling ...", end=" ")
            shuffle_idxs = np.arange(len(arr))
            self.rng.shuffle(shuffle_idxs)
            out = [arr[shuffle_idx] for shuffle_idx in shuffle_idxs]
            print("Done!")
            return out
        else:
            return arr

    def get_largest_file(self):
        """
        Method to get largest media in the dataset.
        returns: largest media element in the dataset.
        """
        return self.max_file

    def get_media_output_type(self):
        """
        Method to specify type of media output produced by the reader.
        returns: type of media output which is produced by this reader.
        """
        return ro.FILE_LIST

    def __len__(self):
        """
        Method to get dataset length.
        returns: length of dataset in units of batch_size.
        """
        return self.num_batches_slice  # self.len

    def __iter__(self):
        """
        Method to initialize iterator.
        """
        self.keys_slice = self.shuffler(self.keys_slice)
        self.current_index = 0
        return self

    def __next__(self):
        """
        Method to get one batch of dataset ouput from iterator.
        """
        last_index = self.current_index + self.batch_size
        # if last_index > (self.batch_size * self.num_batches_slice):
        if last_index > (self.num_keys_slice + len(self.keys_slice_pad)):
            raise StopIteration
        images = []
        ids = np.zeros(shape=self.np_ids_shape, dtype=dt.UINT32)
        input_ids = np.zeros(shape=self.np_inputids_shape, dtype=dt.UINT32)
        attention_mask = np.zeros(shape=self.np_attention_shape, dtype=dt.UINT32)
        batch = np.array(
            [self.batch_size - (0, len(self.keys_slice_pad))[last_index > self.num_keys_slice]], dtype=dt.UINT32
        )
        for i, index in enumerate(range(self.current_index, last_index)):
            image_id, annotkey = self.keys_slice[index - (self.num_keys_slice, 0)[index < self.num_keys_slice]]
            file_name = self.images_slice[image_id]
            image_path = os.path.join(self.root, file_name)
            images.append(image_path)
            ids[i] = image_id
            input_ids[i, :] = self.id_tokenized_caption_map1[(image_id, annotkey)]
            attention_mask[i, :] = self.id_tokenized_caption_map2[(image_id, annotkey)]
        self.current_index = last_index
        images = np.array(images)
        # TODO sasarkar... ids is not needed
        return images, ids, input_ids, attention_mask, batch

    def gen_output_info(self):
        """
        Method to generate output type information.
        :returns : output tensor information of type "opnode_tensor_info".
        """
        dtypes = [dt.NDT, dt.UINT32, dt.UINT32, dt.UINT32, dt.UINT32]
        shapes = [
            self.np_images_shape,
            self.np_ids_shape,
            self.np_inputids_shape,
            self.np_attention_shape,
            self.np_batch_shape,
        ]
        return [opnode_tensor_info(dtype, shp[::-1], "") for dtype, shp in zip(dtypes, shapes)]

    def set_params(self, params):
        """
        Setter method to set mediapipe specific params.
        :params params: mediapipe params of type "opnode_params".
        """
        self.batch_size = params.batch_size
        self.np_images_shape = np.array([self.batch_size], dtype=np.uint32)
        self.np_ids_shape = np.array([self.batch_size], dtype=np.uint32)
        self.np_inputids_shape = np.array([self.batch_size, 128], dtype=np.uint32)
        self.np_attention_shape = np.array([self.batch_size, 128], dtype=np.uint32)  # this is variable output
        self.np_batch_shape = np.array([1], dtype=np.uint32)
        self.round_slice_list(self.batch_size, self.partial_batch)
        if (not self.partial_batch) and (len(self.keys_slice_pad) != 0):
            raise ValueError("expected empty pad key list")
        self.num_batches_slice = int((self.num_keys_slice + len(self.keys_slice_pad)) / self.batch_size)
        # print("coco_reader images {} keys {} batches {}  batchsize {}".format(
        #      len(self.images_slice), self.num_keys_slice, self.num_batches_slice, self.batch_size))
        print("coco_reader batches {} batchsize {}".format(self.num_batches_slice, self.batch_size))

    def round_slice_list(self, round, pad_separate):
        """
        Method to round up/down.
        """
        # this function works on sliced dataset only
        if not self.drop_remainder:
            self.keys_slice, key_slice_pad = roundup_keylist(
                self.rng, self.images_slice, self.keys_slice, round, self.pad_remainder, pad_separate
            )
            if pad_separate:
                self.keys_slice_pad = np.append(self.keys_slice_pad, key_slice_pad)
        else:
            self.images_slice, self.keys_slice = rounddown_keylist(self.images_slice, self.keys_slice, round)
        self.num_keys_slice = len(self.keys_slice)


coco_reader_params["dataset"] = None
schema.add_operator("ClipCocoReader", None, 0, 0, [], 5, coco_reader_params, None, clip_coco_reader, dt.NDT)
op_class = fn.operator_add("ClipCocoReader")
op_class.__module__ = fn.__name__
setattr(fn, "ClipCocoReader", op_class)


class ClipHPUMediaPipe(MediaPipe):
    instance_count = 0

    def __init__(
        self,
        root,
        dataset,
        batch_size,
        shuffle,
        drop_last,
        prefetch_count,
        num_instances,
        instance_id,
        device,
        train_mode,
    ):
        self.super_init = False
        self.root = root
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_instances = num_instances
        self.instance_id = instance_id
        self.ssd_train = train_mode
        self.dataset = dataset

        pipename = "{}:{}".format(self.__class__.__name__, ClipHPUMediaPipe.instance_count)
        pipename = str(pipename)

        super().__init__(device=device, batch_size=batch_size, prefetch_depth=prefetch_count, pipe_name=pipename)
        self.super_init = True
        ClipHPUMediaPipe.instance_count += 1

    def __del__(self):
        if self.super_init:
            super().__del__()

    def definegraph(self):
        seed_mediapipe = 1000
        output_partial = not self.drop_last
        self.input = fn.ClipCocoReader(
            root=self.root,
            seed=seed_mediapipe,
            shuffle=self.shuffle,
            drop_remainder=self.drop_last,
            num_slices=self.num_instances,
            slice_index=self.instance_id,
            partial_batch=output_partial,
            dataset=self.dataset,
        )
        jpegs, ids, inputids, attention, batch = self.input()
        self.decode = fn.ImageDecoder(
            output_format=it.RGB_P,
            resize=[self.dataset.decode_width, self.dataset.decode_height],
            resampling_mode=ft.BICUBIC,
            decoder_stage=ds.ENABLE_ALL_STAGES,
            random_crop_type=rct.CENTER_CROP,
        )
        images = self.decode(jpegs)

        normalize_mean = np.array(self.dataset.mean, dtype=np.float32)
        self.norm_mean = fn.MediaConst(data=normalize_mean, shape=[1, 1, 3], dtype=dt.FLOAT32)
        normalize_std = np.array(self.dataset.std, dtype=np.float32)
        self.norm_std = fn.MediaConst(data=normalize_std, shape=[1, 1, 3], dtype=dt.FLOAT32)

        self.cmn = fn.CropMirrorNorm(
            crop_w=self.dataset.decode_width, crop_h=self.dataset.decode_height, crop_d=0, dtype=dt.FLOAT32
        )
        images = self.cmn(images, self.norm_mean(), self.norm_std())
        return images, ids, inputids, attention, batch


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _get_world_size():
    return dist.get_world_size() if _is_distributed() else 1


def _get_rank():
    return dist.get_rank() if _is_distributed() else 0


class HPUClipPytorchIterator(HPUGenericPytorchIterator):
    """
    Class defining SSD mediapipe iterator for Pytorch framework.
    This class provides functionality to get output tensors from mediapipe.
    """

    def __init__(self, mediapipe):
        """
        Constructor method.
        :params mediapipe: mediapipe
        """
        super().__init__(mediapipe=mediapipe)

    def __next__(self):
        """
        Method to run mediapipe iterator over one batch of dataset and return the output tensors.
        :returns : output tensors.
        """
        # lengths is not returned from iterator
        items_tensors = [self.proxy_device.get_tensor(item.dev_addr) for item in self.pipe.run()]
        b_size = self.pipe.getBatchSize()
        batch = items_tensors[-1].to("cpu").numpy()[0]
        if batch < b_size:
            for i in range(4):
                items_tensors[i] = torch.narrow(items_tensors[i], 0, 0, batch)
        return items_tensors


# using modified coco reader
# TODO.. can this be a subclass of HPUGenericPytorchIterator, and then we remove HPUClipPytorchIterator ?
class ClipHabanaDataLoader:
    def __init__(self, *args, **kwargs):
        def helper(key, loc, default):
            if key in kwargs:
                return kwargs[key]
            else:
                return args[loc] if len(args) >= (loc + 1) else default

        self.dataset = helper("dataset", 0, None)
        self.batch_size = helper("batch_size", 1, 1)
        self.sampler = helper(
            "sampler", 3, None
        )  # TODO sasarkar: pass on sampler and use it for shuffling in mediapipe
        # Cannot pass self.sampler to ClipHPUMediaPipe down to coco reader because it is unpickleable, and doesnt deepcopy
        self.drop_last = helper("drop_last", 8, False)

        imgroot = list({"/".join(imgpath.split("/")[:-1]) for imgpath in self.dataset["image_path"]})
        assert len(imgroot) == 1

        pipeline = ClipHPUMediaPipe(
            root=imgroot[0],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            prefetch_count=3,
            num_instances=_get_world_size(),
            instance_id=_get_rank(),
            device="gaudi2",
            train_mode=True,
            dataset=self.dataset,
        )
        self.iterator = HPUClipPytorchIterator(mediapipe=pipeline)
        self.__iter = None
        print(
            f"Running with Habana media DataLoader with num_instances = {pipeline.num_instances}, instance_id = {pipeline.instance_id}."
        )

    def __iter__(self):
        if self.__iter is None:
            self.__iter = iter(self.iterator)
        return self

    def __next__(self):
        try:
            items = next(self.__iter)
            return {"pixel_values": items[0], "input_ids": items[2], "attention_mask": items[3], "return_loss": True}
        except StopIteration:
            self.__iter = None
            raise

    def __len__(self):
        return len(self.iterator)
