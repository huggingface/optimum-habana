import copy
import importlib
import json
import logging
import math
import os
import queue
import tempfile
import traceback
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import torch
import torch.multiprocessing as mp
import transformers
from huggingface_hub import HfApi
from numpy import ndarray
from torch import Tensor, device, nn
from tqdm.autonotebook import trange
from transformers import is_torch_npu_available

from sentence_transformers.model_card import SentenceTransformerModelCardData, generate_model_card
from sentence_transformers.similarity_functions import SimilarityFunction

from sentence_transformers import __MODEL_HUB_ORGANIZATION__, __version__
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.fit_mixin import FitMixin
from sentence_transformers.models import Normalize, Pooling, Transformer
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.util import (
    batch_to_device,
    get_device_name,
    import_from_string,
    is_sentence_transformer_model,
    load_dir_path,
    load_file_path,
    save_to_hub_args_decorator,
    truncate_embeddings,
)

logger = logging.getLogger(__name__)


def st_gaudi_encode(
    self,
    sentences: Union[str, List[str]],
    prompt_name: Optional[str] = None,
    prompt: Optional[str] = None,
    batch_size: int = 32,
    show_progress_bar: bool = None,
    output_value: Optional[Literal["sentence_embedding", "token_embeddings"]] = "sentence_embedding",
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
    convert_to_numpy: bool = True,
    convert_to_tensor: bool = False,
    device: str = None,
    normalize_embeddings: bool = False,
) -> Union[List[Tensor], ndarray, Tensor]:
    """
    Computes sentence embeddings.

    Args:
        sentences (Union[str, List[str]]): The sentences to embed.
        prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
            which is either set in the constructor or loaded from the model configuration. For example if
            ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
            is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
            is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
        prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
            sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
            because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
        batch_size (int, optional): The batch size used for the computation. Defaults to 32.
        show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.
        output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return:
            "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,
            to get all output values. Defaults to "sentence_embedding".
        precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
            Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.
            Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for
            reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
        convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
            Defaults to True.
        convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
            Defaults to False.
        device (str, optional): Which :class:`torch.device` to use for the computation. Defaults to None.
        normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
            the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.

    Returns:
        Union[List[Tensor], ndarray, Tensor]: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.
        If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If ``convert_to_tensor``,
        a torch Tensor is returned instead. If ``self.truncate_dim <= output_dimension`` then output_dimension is ``self.truncate_dim``.

    Example:
        ::

            from sentence_transformers import SentenceTransformer

            # Load a pre-trained SentenceTransformer model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Encode some texts
            sentences = [
                "The weather is lovely today.",
                "It's so sunny outside!",
                "He drove to the stadium.",
            ]
            embeddings = model.encode(sentences)
            print(embeddings.shape)
            # (3, 768)
    """
    self.eval()
    if show_progress_bar is None:
        show_progress_bar = (
            logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
        )

    if convert_to_tensor:
        convert_to_numpy = False

    if output_value != "sentence_embedding":
        convert_to_tensor = False
        convert_to_numpy = False

    input_was_string = False
    if isinstance(sentences, str) or not hasattr(
        sentences, "__len__"
    ):  # Cast an individual sentence to a list with length 1
        sentences = [sentences]
        input_was_string = True

    if prompt is None:
        if prompt_name is not None:
            try:
                prompt = self.prompts[prompt_name]
            except KeyError:
                raise ValueError(
                    f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(self.prompts.keys())!r}."
                )
        elif self.default_prompt_name is not None:
            prompt = self.prompts.get(self.default_prompt_name, None)
    else:
        if prompt_name is not None:
            logger.warning(
                "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                "Ignoring the `prompt_name` in favor of `prompt`."
            )
    extra_features = {}
    if prompt is not None:
        sentences = [prompt + sentence for sentence in sentences]

        # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
        # Tracking the prompt length allow us to remove the prompt during pooling
        tokenized_prompt = self.tokenize([prompt])
        if "input_ids" in tokenized_prompt:
            extra_features["prompt_length"] = tokenized_prompt["input_ids"].shape[-1] - 1

    if device is None:
        device = self.device

    all_embeddings = []
    length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]


    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
        sentences_batch = sentences_sorted[start_index : start_index + batch_size]
        features = self.tokenize(sentences_batch)
        
        if self.device.type == "hpu":
            if "input_ids" in features:
                curr_tokenize_len = features["input_ids"].shape
                additional_pad_len = 2 ** math.ceil(math.log2(curr_tokenize_len[1])) - curr_tokenize_len[1]
                features["input_ids"] = torch.cat(
                    (
                        features["input_ids"],
                        torch.ones((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                    ),
                    -1,
                )
                features["attention_mask"] = torch.cat(
                    (
                        features["attention_mask"],
                        torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                    ),
                    -1,
                )
                if "token_type_ids" in features:
                    features["token_type_ids"] = torch.cat(
                        (
                            features["token_type_ids"],
                            torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                        ),
                        -1,
                    )
        features = batch_to_device(features, device)
        features.update(extra_features)

        with torch.no_grad():
            out_features = self.forward(features)
            if self.device.type == "hpu":
                out_features = copy.deepcopy(out_features)

            out_features["sentence_embedding"] = truncate_embeddings(
                out_features["sentence_embedding"], self.truncate_dim
            )

            if output_value == "token_embeddings":
                embeddings = []
                for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                    last_mask_id = len(attention) - 1
                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1

                    embeddings.append(token_emb[0 : last_mask_id + 1])
            elif output_value is None:  # Return all outputs
                embeddings = []
                for sent_idx in range(len(out_features["sentence_embedding"])):
                    row = {name: out_features[name][sent_idx] for name in out_features}
                    embeddings.append(row)
            else:  # Sentence embeddings
                embeddings = out_features[output_value]
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

    if precision and precision != "float32":
        all_embeddings = quantize_embeddings(all_embeddings, precision=precision)

    if convert_to_tensor:
        if len(all_embeddings):
            if isinstance(all_embeddings, np.ndarray):
                all_embeddings = torch.from_numpy(all_embeddings)
            else:
                all_embeddings = torch.stack(all_embeddings)
        else:
            all_embeddings = torch.Tensor()
    elif convert_to_numpy:
        if not isinstance(all_embeddings, np.ndarray):
            if all_embeddings[0].dtype == torch.bfloat16:
                all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
            else:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
    elif isinstance(all_embeddings, np.ndarray):
        all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

    if input_was_string:
        all_embeddings = all_embeddings[0]

    return all_embeddings
    
