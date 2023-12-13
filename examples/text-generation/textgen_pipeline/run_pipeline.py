import argparse

from pipeline import GaudiTextGenerationPipeline


def setup_parser(parser):
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature value for text generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top_p value for sampling output probabilities.")
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="Repetition penalty for text generation."
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="Whether to perfrom sampling instead of greedy decoding."
    )
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for text generation.")

    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)

    if args.do_sample:
        pipe = GaudiTextGenerationPipeline(
            model_name_or_path=args.model_name_or_path,
            bf16=args.bf16,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
        )
    else:
        pipe = GaudiTextGenerationPipeline(
            model_name_or_path=args.model_name_or_path,
            bf16=args.bf16,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
    pipe.compile_graph()

    print(f"Prompt: {args.prompt}")
    for i in range(10):
        print(f"Response {i+1}: {pipe(args.prompt)}")


if __name__ == "__main__":
    main()
