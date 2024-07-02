import warnings

import packaging.version
import torch
import transformers
from peft import PeftType


def gaudi_generate(self, *args, **kwargs):
    peft_config = self.active_peft_config
    self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
    if hasattr(self.base_model, "model"):
        self.base_model.model.generation_config = self.generation_config
    else:
        self.base_model.generation_config = self.generation_config
    try:
        if not peft_config.is_prompt_learning:
            with self._enable_peft_forward_hooks(*args, **kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                outputs = self.base_model.generate(*args, **kwargs)
        else:
            kwargs["num_virtual_tokens"] = peft_config.num_virtual_tokens
            outputs = self.base_model.generate(**kwargs)
    except:
        self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        raise
    else:
        self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        return outputs


def gaudi_prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
    """
    Copied from PeftModelForCausalLM.prepare_inputs_for_generation: https://github.com/huggingface/peft/blob/v0.9.0/src/peft/peft_model.py#L1156
    The only differences are:
    - add token_idx disposal for prompt learning
    """
    peft_config = self.active_peft_config
    model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
    # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
    # for some architectures which requires a special fix for prompt tuning etc.
    # TODO: starting with transformers 4.38, all architectures should support caching.
    uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
    uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
    transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
    uses_cache = uses_transformers_4_38 or (
        uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
    )
    if peft_config.peft_type == PeftType.POLY:
        model_kwargs["task_ids"] = task_ids

    if peft_config.is_prompt_learning:
        if uses_cache and (model_kwargs["past_key_values"] is not None):
            # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
            # In prompt learning methods, past key values are longer when compared to the `input_ids`.
            # As such only consider the last input ids in the autogressive generation phase.
            if model_kwargs.get("reuse_cache", False):
                if model_kwargs["past_key_values"][0][0][-2] >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]
            else:
                if model_kwargs["past_key_values"][0][0].shape[-2] >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

        if model_kwargs.get("attention_mask", None) is not None:
            size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
            prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
            model_kwargs["attention_mask"] = torch.cat((prefix_attention_mask, model_kwargs["attention_mask"]), dim=1)

        token_idx = model_kwargs.get("token_idx", None)
        if token_idx is not None:
            token_idx = token_idx + peft_config.num_virtual_tokens

        token_idx_cpu = model_kwargs.get("token_idx_cpu", None)
        if token_idx_cpu is not None:
            token_idx_cpu = token_idx_cpu + peft_config.num_virtual_tokens
            model_kwargs["token_idx_cpu"] = token_idx_cpu

        if model_kwargs.get("position_ids", None) is not None and token_idx is None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            model_kwargs["position_ids"] = None

        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None

        if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
            model_kwargs["past_key_values"] = past_key_values
        else:
            if model_kwargs["past_key_values"] is None:
                inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                prompts = prompts.to(inputs_embeds.dtype)
                model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                model_kwargs["input_ids"] = None
        if token_idx is not None:
            attention_mask = model_kwargs["attention_mask"]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if peft_config.peft_type == PeftType.PREFIX_TUNING and model_kwargs["input_ids"].shape[-1] != 1:
                position_ids = position_ids[:, -model_kwargs["input_ids"].shape[-1] :]
            if model_kwargs["past_key_values"] is not None and model_kwargs["input_ids"].shape[-1] == 1:
                position_ids = torch.index_select(position_ids, 1, token_idx - 1)
            model_kwargs["position_ids"] = position_ids
            model_kwargs["token_idx"] = token_idx
    # For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is
    # passed in the forward pass to keep track of the position ids of the cache. We have to
    # pop that from `model_kwargs` as `cache_position` is properly created by the model, using the passed
    # `inputs_embeds`: https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956
    _ = model_kwargs.pop("cache_position", None)

    return model_kwargs
