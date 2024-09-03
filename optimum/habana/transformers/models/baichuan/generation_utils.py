# Copyright 2023 Baichuan Inc. All Rights Reserved.

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
Adapted from the following sources:
https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/generation_utils.py
https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/main/generation_utils.py
"""

from queue import Queue
from typing import List

import torch


def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int = 0):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return torch.LongTensor([input_tokens]).to(model.device)


class TextIterStreamer:
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            self.text_queue.put(self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens))

    def end(self):
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value
