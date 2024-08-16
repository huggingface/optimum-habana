#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Intel Corporation and The HuggingFace Team. All rights reserved.
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

import argparse
import logging
import math
import time

from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from pipeline import GaudiTextGenerationPipeline
from run_generation import setup_parser


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)

    # Initialize the pipeline
    pipe = GaudiTextGenerationPipeline(args, logger, use_with_langchain=True, warmup_on_init=False)

    # Create LangChain object
    hf = HuggingFacePipeline(pipeline=pipe)

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    Context: Large Language Models (LLMs) are the latest models used in NLP.
    Their superior performance over smaller models has made them incredibly
    useful for developers building NLP enabled applications. These models
    can be accessed via Hugging Face's `transformers` library, via OpenAI
    using the `openai` library, and via Cohere using the `cohere` library.

    Question: {question}
    Answer: """

    prompt = PromptTemplate(input_variables=["question"], template=template)
    chain = prompt | hf

    questions = [
        {"question": "Which libraries and model providers offer LLMs?"},
        {"question": "What is the provided context about?"},
        {"question": "Can I use LLMs on CPU?"},
        {"question": "How easy is to build my own LLM?"},
        {"question": "Can I use LLM to order pizza?"},
        {"question": "Can I install LLM into my phone?"},
    ]

    if args.batch_size > len(questions):
        times_to_extend = math.ceil(args.batch_size / len(questions))
        questions = questions * times_to_extend

    input_questions = questions[: args.batch_size]

    import habana_frameworks.torch.hpu as torch_hpu

    logger.info("LangChain warmup (graph compilation)...")
    for _ in range(args.warmup):
        _ = chain.batch(input_questions)
    torch_hpu.synchronize()

    duration = 0
    for iteration in range(args.n_iterations):
        t0 = time.perf_counter()
        responses = chain.batch(input_questions)
        duration += time.perf_counter() - t0

        for i, (question, answer) in enumerate(zip(input_questions, responses)):
            print(f"Question[{iteration+1}][{i+1}]: {question['question']}")
            print(f"Response[{iteration+1}][{i+1}]: {answer}\n")

    throughput = args.n_iterations * args.batch_size * args.max_new_tokens / duration
    print(f"Inference Duration (for {args.n_iterations} iterations): {duration} seconds")
    print(f"Throughput: {throughput} tokens/second")


if __name__ == "__main__":
    main()
