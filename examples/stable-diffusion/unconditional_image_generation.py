import argparse
import logging
import sys

from diffusers import DDPMScheduler
from transformers.utils import check_min_version

from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiDDPMPipeline
from optimum.habana.transformers.gaudi_configuration import GaudiConfig


logger = logging.getLogger(__name__)

try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


check_min_version("4.51.0")
check_optimum_habana_min_version("1.18.0.dev0")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="google/ddpm-ema-celebahq-256",
        type=str,
        help="Path of the pre-trained unconditional image generation model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for the task.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=1000, help="Number of inference steps for the denoising UNet."
    )
    parser.add_argument(
        "--use_gaudi_ddim_scheduler",
        action="store_true",
        help="Whether to use the Gaudi optimized DDIM scheduler. The default is DDPMScheduler",
    )
    parser.add_argument(
        "--use_habana",
        action="store_true",
        help="Whether to use HPU for computations.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 precision for classification.",
    )
    parser.add_argument(
        "--sdp_on_bf16",
        action="store_true",
        default=False,
        help="Allow pyTorch to use reduced precision in the SDPA math backend",
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Whether to save the generated images to jpg.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/",
        help="Where to save the generated images. The default is DDPMScheduler.",
    )
    parser.add_argument(
        "--throughput_warmup_steps",
        type=int,
        default=3,
        help="Number of steps to ignore for throughput calculation.",
    )

    args = parser.parse_args()
    model_name = args.model_name_or_path

    if args.use_gaudi_ddim_scheduler:
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name)
    else:
        scheduler = DDPMScheduler.from_pretrained(model_name)

    gaudi_kwargs = {
        "use_torch_autocast": args.bf16,
    }
    gaudi_config = GaudiConfig(**gaudi_kwargs)

    kwargs = {
        "scheduler": scheduler,
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": gaudi_config,
        "sdp_on_bf16": args.sdp_on_bf16,
    }

    kwargs_call = {"throughput_warmup_steps": args.throughput_warmup_steps}

    pipeline = GaudiDDPMPipeline.from_pretrained(model_name, **kwargs)
    output = pipeline(batch_size=args.batch_size, num_inference_steps=args.num_inference_steps, **kwargs_call)

    if args.output_dir:
        logger.info(f"Generating outputs to {args.output_dir}")
        for i in range(len(output.images)):
            output.images[i].save(args.output_dir + "unconditional_image_" + str(i) + ".jpg")


if __name__ == "__main__":
    main()
