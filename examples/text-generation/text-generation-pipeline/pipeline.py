import os
import sys

import torch
from transformers import TextGenerationPipeline


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class GaudiTextGenerationPipeline(TextGenerationPipeline):
    def __init__(self, args, logger, use_with_langchain=False, warmup_on_init=True):
        from utils import initialize_model

        self.model, _, self.tokenizer, self.generation_config = initialize_model(args, logger)

        self.task = "text-generation"
        self.device = args.device

        if args.do_sample:
            self.generation_config.temperature = args.temperature
            self.generation_config.top_p = args.top_p

        self.max_padding_length = args.max_input_tokens if args.max_input_tokens > 0 else 100
        self.use_hpu_graphs = args.use_hpu_graphs
        self.profiling_steps = args.profiling_steps
        self.profiling_warmup_steps = args.profiling_warmup_steps
        self.profiling_record_shapes = args.profiling_record_shapes

        self.use_with_langchain = use_with_langchain
        if self.use_with_langchain:
            self.generation_config.ignore_eos = False

        if warmup_on_init:
            import habana_frameworks.torch.hpu as torch_hpu

            logger.info("Graph compilation...")

            warmup_promt = ["Here is my prompt"] * args.batch_size
            for _ in range(args.warmup):
                _ = self(warmup_promt)
            torch_hpu.synchronize()

    def __call__(self, prompt):
        use_batch = isinstance(prompt, list)

        if use_batch:
            model_inputs = self.tokenizer.batch_encode_plus(
                prompt, return_tensors="pt", max_length=self.max_padding_length, padding="max_length", truncation=True
            )
        else:
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
            hpu_graphs=self.use_hpu_graphs,
            profiling_steps=self.profiling_steps,
            profiling_warmup_steps=self.profiling_warmup_steps,
            profiling_record_shapes=self.profiling_record_shapes,
        ).cpu()

        if use_batch:
            output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if self.use_with_langchain:
            if use_batch:
                return [{"generated_text": unbatched_output_text} for unbatched_output_text in output_text]
            else:
                return [{"generated_text": output_text}]

        return output_text
