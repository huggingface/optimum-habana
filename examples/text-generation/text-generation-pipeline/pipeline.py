import torch
from transformers import TextGenerationPipeline
from utils import initialize_model


class GaudiTextGenerationPipeline(TextGenerationPipeline):
    def __init__(self, args, logger, use_with_langchain=False):
        self.model, self.tokenizer, self.generation_config = initialize_model(args, logger)

        self.task = "text-generation"
        self.device = args.device

        if args.do_sample:
            self.generation_config.temperature = args.temperature
            self.generation_config.top_p = args.top_p

        self.max_padding_length = args.max_input_tokens if args.max_input_tokens > 0 else 100
        self.use_hpu_graphs = args.use_hpu_graphs
        self.profiling_steps = args.profiling_steps
        self.profiling_warmup_steps = args.profiling_warmup_steps

        self.use_with_langchain = use_with_langchain
        if self.use_with_langchain:
            self.generation_config.ignore_eos = False

        import habana_frameworks.torch.hpu as torch_hpu

        logger.info("Graph compilation...")
        for _ in range(3):
            self("Here is my prompt")
        torch_hpu.synchronize()

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
            hpu_graphs=self.use_hpu_graphs,
            profiling_steps=self.profiling_steps,
            profiling_warmup_steps=self.profiling_warmup_steps,
        ).cpu()

        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if self.use_with_langchain:
            return [{"generated_text": output_text}]

        return output_text
