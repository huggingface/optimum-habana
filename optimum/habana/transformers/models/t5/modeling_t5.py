from typing import Optional

import torch
from transformers.deepspeed import is_deepspeed_zero3_enabled


try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm as FusedRMSNorm
except ImportError:
    print("Not using HPU fused kernel for RMSNorm")
    FusedRMSNorm = None


def gaudi_t5_layernorm_forward(self, hidden_states):
    """
    Copied from T5LayerNorm.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
    The only differences are:
        - override RMSNorm with Habana fused RMSNorm
    """
    if not self.training and hidden_states.device.type == "hpu" and FusedRMSNorm:
        orig_dtype = hidden_states.dtype
        hidden_states = FusedRMSNorm.apply(hidden_states.float(), self.weight.float(), self.variance_epsilon)
        return hidden_states.to(orig_dtype)
    else:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


def _gaudi_get_resized_embeddings(
    self, old_embeddings: torch.nn.Embedding, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None
) -> torch.nn.Embedding:
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/modeling_utils.py#L1424
    """
    if new_num_tokens is None:
        return old_embeddings

    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    else:
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

    if old_num_tokens == new_num_tokens:
        return old_embeddings

    if not isinstance(old_embeddings, torch.nn.Embedding):
        raise TypeError(
            f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {torch.nn.Embedding}. You"
            " should either use a different resize function or make sure that `old_embeddings` are an instance of"
            f" {torch.nn.Embedding}."
        )

    # Build new embeddings
    new_embeddings = torch.nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

    # initialize all new embeddings (in particular added tokens)
    self._init_weights(new_embeddings)

    # Copy token embeddings from the previous weights

    # numbers of tokens to copy
    n = min(old_num_tokens, new_num_tokens)
    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    else:
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

    return new_embeddings


def _gaudi_get_resized_lm_head(
    self, old_lm_head: torch.nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
) -> torch.nn.Linear:
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/modeling_utils.py#L1488
    """
    if new_num_tokens is None:
        return old_lm_head

    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )
    else:
        old_num_tokens, old_lm_head_dim = (
            old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        )

    if old_num_tokens == new_num_tokens:
        return old_lm_head

    if not isinstance(old_lm_head, torch.nn.Linear):
        raise TypeError(
            f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {torch.nn.Linear}. You"
            " should either use a different resize function or make sure that `old_lm_head` are an instance of"
            f" {torch.nn.Linear}."
        )

    # Build new lm head
    new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
    has_new_lm_head_bias = old_lm_head.bias is not None
    new_lm_head = torch.nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias)
    new_lm_head = new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)

    # initialize new lm head (in particular added tokens)
    self._init_weights(new_lm_head)

    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

    # XXX: put the long block of code in a wrapper
    if is_deepspeed_zero3_enabled():
        import deepspeed

        params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                # Copy old lm head weights to new lm head
                if not transposed:
                    new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
                else:
                    new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

                # Copy bias weights to new lm head
                if has_new_lm_head_bias:
                    new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
    else:
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

    return new_lm_head
