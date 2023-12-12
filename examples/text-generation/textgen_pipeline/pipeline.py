import copy
import os

import habana_frameworks.torch.hpu as torch_hpu
import torch
from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
)

from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from optimum.habana.utils import set_seed


class GaudiTextGenerationPipeline(TextGenerationPipeline):
    """
    An end-to-end text-generation pipeline that can used to initialize LangChain classes. It supports both single-hpu and multi-hpu inference.
    """

    def __init__(self, model_name_or_path=None, use_bf16=True, **kwargs):
        self.use_deepspeed = "deepspeed" in os.environ["_"]

        if self.use_deepspeed:
            world_size, _, self.local_rank = initialize_distributed_hpu()

            import deepspeed

            # Initialize Deepspeed processes
            deepspeed.init_distributed(dist_backend="hccl")

        self.task = "text-generation"
        self.device = "hpu"

        # Tweak generation so that it runs faster on Gaudi
        adapt_transformers_to_gaudi()
        set_seed(27)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if self.use_deepspeed or use_bf16:
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float

        if self.use_deepspeed:
            config = AutoConfig.from_pretrained(model_name_or_path)
            is_optimized = model_is_optimized(config)
            load_to_meta = model_on_meta(config)

            if load_to_meta:
                # Construct model with fake meta tensors, later will be replaced on devices during ds-inference ckpt load
                with deepspeed.OnDevice(dtype=model_dtype, device="meta"):
                    model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)
            else:
                get_repo_root(model_name_or_path, local_rank=self.local_rank)
                # placement on cpu if meta tensors are not supported
                with deepspeed.OnDevice(dtype=model_dtype, device="cpu"):
                    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=model_dtype)
            model = model.eval()

            # Initialize the model
            ds_inference_kwargs = {"dtype": model_dtype}
            ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
            ds_inference_kwargs["enable_cuda_graph"] = True

            if load_to_meta:
                # model loaded to meta is managed differently
                checkpoints_json = "checkpoints.json"
                write_checkpoints_json(model_name_or_path, self.local_rank, checkpoints_json)

            # Make sure all devices/nodes have access to the model checkpoints
            torch.distributed.barrier()

            ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(config)
            if load_to_meta:
                ds_inference_kwargs["checkpoint"] = checkpoints_json

            model = deepspeed.init_inference(model, **ds_inference_kwargs)
            model = model.module
        else:
            get_repo_root(model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=model_dtype)
            model = model.eval().to(self.device)
            is_optimized = model_is_optimized(model.config)
            model = wrap_in_hpu_graph(model)

        self.model = model

        # Used for padding input to fixed length
        self.tokenizer.padding_side = "left"
        self.max_padding_length = kwargs.get("max_padding_length", self.model.config.max_position_embeddings)

        # Define config params for llama models
        if self.model.config.model_type == "llama":
            self.model.generation_config.pad_token_id = 0
            self.model.generation_config.bos_token_id = 1
            self.model.generation_config.eos_token_id = 2
            self.tokenizer.bos_token_id = self.model.generation_config.bos_token_id
            self.tokenizer.eos_token_id = self.model.generation_config.eos_token_id
            self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
            self.tokenizer.pad_token = self.tokenizer.decode(self.tokenizer.pad_token_id)
            self.tokenizer.eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
            self.tokenizer.bos_token = self.tokenizer.decode(self.tokenizer.bos_token_id)

        # Applicable to models that do not have pad tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        # Edit generation configuration based on input arguments
        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.max_new_tokens = kwargs.get("max_new_tokens", 100)
        self.generation_config.use_cache = kwargs.get("use_kv_cache", True)
        self.generation_config.static_shapes = is_optimized
        self.generation_config.do_sample = kwargs.get("do_sample", False)
        self.generation_config.num_beams = kwargs.get("num_beams", 1)
        self.generation_config.temperature = kwargs.get("temperature", 1.0)
        self.generation_config.top_p = kwargs.get("top_p", 1.0)
        self.generation_config.repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        self.generation_config.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.generation_config.bad_words_ids = None
        self.generation_config.force_words_ids = None
        self.generation_config.ignore_eos = False

        if self.use_deepspeed:
            torch.distributed.barrier()

    def __call__(self, prompt: str):
        model_inputs = self.tokenizer.encode_plus(
            prompt, return_tensors="pt", max_length=self.max_padding_length, padding="max_length", truncation=True
        )

        for t in model_inputs:
            if torch.is_tensor(model_inputs[t]):
                model_inputs[t] = model_inputs[t].to(self.device)

        output = self.model.generate(
            **model_inputs,
            generation_config=self.generation_config,
            lazy_mode=True,
            hpu_graphs=True,
            profiling_steps=0,
            profiling_warmup_steps=0,
        ).cpu()

        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        del output, model_inputs

        return [{"generated_text": output_text}]

    def get_process_rank(self):
        """
        Function that returns process id during distributed inference.
        """
        if self.use_deepspeed:
            return self.local_rank
        return -1

    def compile_graph(self):
        """
        Function to compile computation graphs and synchronize hpus.
        """
        for _ in range(3):
            self("Here is my prompt")
        torch_hpu.synchronize()


def main():
    pipe = GaudiTextGenerationPipeline(
        model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
        max_new_tokens=100,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
    )
    pipe.compile_graph()

    # Test model on different input prompts
    print("Test 1: short prompt")
    print(pipe("Once upon a time"))
    print("Success!\n")

    print("Test 2: long prompt")
    print(
        pipe(
            "Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them"
        )
    )
    print("Success!\n")

    print("Test 3: qa prompt")
    print(
        pipe(
            """Answer the question based on the context below. If the question cannot be answered using the information provided answer with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP. Their superior performance over smaller models has made them incredibly useful for developers building NLP enabled applications. These models can be accessed via Hugging Face's `transformers` library, via OpenAI using the `openai` library, and via Cohere using the `cohere` library.

Question: Which libraries and model providers offer LLMs?

Answer: """
        )
    )
    print("Success!")


if __name__ == "__main__":
    main()
