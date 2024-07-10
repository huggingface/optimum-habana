# This script is based on https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/biomed_clip_example.ipynb
import argparse
import json
import logging
import os
import time
from pathlib import Path
from pprint import pprint
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy
import torch
from open_clip import create_model_from_pretrained, get_tokenizer, model
from PIL import Image

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATASET_URL = "https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/"
LABELS = [
    "adenocarcinoma histopathology",
    "brain MRI",
    "covid line chart",
    "squamous cell carcinoma histopathology",
    "immunohistochemistry histopathology",
    "bone X-ray",
    "chest X-ray",
    "pie chart",
    "hematoxylin and eosin histopathology",
]

TEST_IMGS = [
    "squamous_cell_carcinoma_histopathology.jpeg",
    "H_and_E_histopathology.jpg",
    "bone_X-ray.jpg",
    "adenocarcinoma_histopathology.jpg",
    "covid_line_chart.png",
    "IHC_histopathology.jpg",
    "chest_X-ray.jpg",
    "brain_MRI.jpg",
    "pie_chart.png",
]


def plot_images_with_metadata(images: list, metadata, output_dir: str, plot_name: str) -> None:
    print(f"plottypes {type(images)} {type(metadata)} {type(output_dir)} {type(plot_name)}")

    num_images = len(images)
    fig, axes = plt.subplots(nrows=num_images, ncols=1, figsize=(5, 5 * num_images))

    for i, (img_path, metadata) in enumerate(zip(images, metadata)):
        img = Image.open(urlopen(img_path))
        if isinstance(axes, list) or isinstance(axes, numpy.ndarray):
            ax = axes[i]
        else:
            ax = axes
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{metadata['filename']}\n{metadata['top_probs']}", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{plot_name}.png")


def run_qa(model: model, images: torch.Tensor, texts: torch.Tensor, device: torch.device) -> tuple:
    with torch.no_grad():
        image_features, text_features, logit_scale = model(images, texts)
        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    return sorted_indices, logits


def postprocess(args: argparse.Namespace, sorted_indices: torch.Tensor, logits: torch.Tensor, topk: int) -> list:
    logits = logits.float().cpu().numpy()
    sorted_indices = sorted_indices.int().cpu().numpy()
    metadata_list = []
    for i, img in enumerate(args.image_path):
        img_name = img.split("/")[-1]

        top_probs = []
        topk = len(args.labels) if topk == -1 else topk
        for j in range(topk):
            jth_index = sorted_indices[i][j]
            top_probs.append(f"{args.labels[jth_index]}: {logits[i][jth_index] * 100:.1f}")

        metadata = {"filename": img_name, "top_probs": "\n".join(top_probs)}
        metadata_list.append(metadata)
    return metadata_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--image_path",
        default=[DATASET_URL + img for img in TEST_IMGS],
        type=str,
        nargs="*",
        help='Path to image as input. Can be a single string (eg: --image_path "URL1"), or a list of space-separated strings (eg: --image_path "URL1" "URL2")',
    )
    parser.add_argument(
        "--topk",
        default=1,
        type=int,
        help="topk num. Provides top K probabilities for the labels provided.",
    )
    parser.add_argument(
        "--prompt",
        default="this is a picture of ",
        type=str,
        help='Prompt for classification. It should be a string separated by comma. (eg: --prompt "a photo of ")',
    )
    parser.add_argument(
        "--labels",
        default=LABELS,
        type=str,
        nargs="*",
        help='Labels for classification (eg: --labels "LABEL1"), or a list of space-separated strings (eg: --labels "LABEL1" "LABEL2")',
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform in bf16 precision.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.getcwd(),
        type=str,
        help="Output directory to store results in.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument(
        "--n_iterations", type=int, default=10, help="Number of inference iterations for benchmarking."
    )
    parser.add_argument("--plot_images", action="store_true", help="Plot images with metadata for verification")
    parser.add_argument(
        "--plot_name",
        default="openclip_vqa_plot",
        type=str,
        help="Name of the plot generated with the image and corresponding top K results",
    )
    parser.add_argument(
        "--print_result",
        action="store_true",
        help="Whether to print the zero shot classification results.",
    )

    args = parser.parse_args()

    adapt_transformers_to_gaudi()

    precision = "fp32"
    dtype = torch.float32
    if args.bf16:
        precision = "bf16"
        dtype = torch.bfloat16

    model, preprocess = create_model_from_pretrained(f"hf-hub:{args.model_name_or_path}", precision=precision)
    tokenizer = get_tokenizer(f"hf-hub:{args.model_name_or_path}")

    device = torch.device("hpu") if torch.hpu.is_available() else torch.device("cpu")
    device_type = "hpu" if torch.hpu.is_available() else "cpu"

    # Initialize model
    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        model = wrap_in_hpu_graph(model)
    model = model.to(device)
    model.eval()

    images = torch.stack([preprocess(Image.open(urlopen(img))) for img in args.image_path]).to(device)
    texts = tokenizer([args.prompt + l for l in args.labels]).to(device)

    # Warm up
    logger.info("Running warmup")
    for i in range(args.warmup):
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=True):
            _, _ = run_qa(model, images, texts, device=device)

    logger.info("Running inference")
    start = time.time()
    for i in range(args.n_iterations):
        logits = None
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=True):
            sorted_indices, logits = run_qa(model, images, texts, device=device)
    end = time.time()

    # Results and metrics
    metadata_list = []
    metadata_list = postprocess(args, sorted_indices, logits, args.topk)
    if args.print_result:
        logger.info("Results from the last iteration:")
        pprint(metadata_list)
    inference_time_per_iteration = (end - start) * 1000 / args.n_iterations
    logger.info(f"Inference Time per iteration = {inference_time_per_iteration:.4}ms")
    throughput = len(args.image_path) * args.n_iterations / (end - start)
    logger.info(f"Throughput = {throughput:.4} images/s")

    # Store results if necessary
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"throughput": throughput, "inference time per iteration ": inference_time_per_iteration}
        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    if args.plot_images:
        plot_images_with_metadata(args.image_path, metadata_list, args.output_dir, args.plot_name)


if __name__ == "__main__":
    main()
