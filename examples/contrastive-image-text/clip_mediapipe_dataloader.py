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

import torch

from optimum.utils import logging


logger = logging.get_logger(__name__)


class MediaApiDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        sampler=None,
        collate_fn=None,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=None,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.fallback_activated = False

        try:
            from clip_media_pipe import ClipMediaPipe
            from habana_frameworks.mediapipe.plugins.iterator_pytorch import HPUGenericPytorchIterator

            pipeline = ClipMediaPipe(
                dataset=dataset,
                sampler=sampler,
                batch_size=batch_size,
                drop_last=drop_last,
                queue_depth=3,
            )
            self.iterator = HPUGenericPytorchIterator(mediapipe=pipeline)
        except Exception as e:
            logger.warning(f"Using Pytorch native dataloader because: {e}.")
            self.fallback_activated = True
            dataset.set_transform(dataset.transform_func)
            super(MediaApiDataLoader, self).__init__(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=worker_init_fn,
            )

    def __len__(self):
        if self.fallback_activated:
            return super().__len__()
        else:
            return len(self.iterator)

    def __iter__(self):
        if self.fallback_activated:
            return super().__iter__()
        else:
            self.iterator.__iter__()
        return self

    def __next__(self):
        if self.fallback_activated:
            return super().__next__()

        data = next(self.iterator)
        return {
            "pixel_values": data[0],
            "input_ids": data[1],
            "attention_mask": data[2],
            "return_loss": True,
        }
