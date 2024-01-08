import argparse
import logging
import time

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
    args.num_return_sequences = 1

    if args.prompt:
        input_sentences = args.prompt
    else:
        input_sentences = [
            "DeepSpeed is a machine learning framework",
            "He is working on",
            "He has a",
            "He got all",
            "Everyone is happy and I can",
            "The new movie that got Oscar this year",
            "In the far far distance from our galaxy,",
            "Peace is the only way",
        ]

    logger.info("Initializing text-generation pipeline...")
    pipe = GaudiTextGenerationPipeline(args, logger)

    logger.info("Running inference...")
    for input_sentence in input_sentences:
        print(f"Prompt: {input_sentence}")
        t0 = time.perf_counter()
        output = pipe(input_sentence)
        duration = time.perf_counter() - t0
        throughput = args.max_new_tokens / duration
        print(f"Generated Text: {repr(output)}")
        print(f"Inference Duration: {duration} seconds")
        print(f"Throughput: {throughput} tokens/second")


if __name__ == "__main__":
    main()
