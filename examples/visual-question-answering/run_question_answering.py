import argparse
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import torch
from urllib.request import urlopen
from PIL import Image
import matplotlib.pyplot as plt
import habana_frameworks.torch.core as htcore
import logging
#from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
import time
from pprint import pprint


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATASET_URL = 'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/'
LABELS = [
        'adenocarcinoma histopathology',
        'brain MRI',
        'covid line chart',
        'squamous cell carcinoma histopathology',
        'immunohistochemistry histopathology',
        'bone X-ray',
        'chest X-ray',
        'pie chart',
        'hematoxylin and eosin histopathology'
    ]

TEST_IMGS = [
        'squamous_cell_carcinoma_histopathology.jpeg',
        'H_and_E_histopathology.jpg',
        'bone_X-ray.jpg',
        'adenocarcinoma_histopathology.jpg',
        'covid_line_chart.png',
        'IHC_histopathology.jpg',
        'chest_X-ray.jpg',
        'brain_MRI.jpg',
        'pie_chart.png'
    ]


def plot_images_with_metadata(images, metadata):
    num_images = len(images)
    fig, axes = plt.subplots(nrows=num_images, ncols=1, figsize=(5, 5 * num_images))

    for i, (img_path, metadata) in enumerate(zip(images, metadata)):
        img = Image.open(urlopen(DATASET_URL + img_path))
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{metadata['filename']}\n{metadata['top_probs']}", fontsize=14)

    plt.tight_layout()
    plt.savefig('foo.png')


def run_qa(model, preprocess, tokenizer, template, device):
    context_length = 256
    images = torch.stack([preprocess(Image.open(urlopen(DATASET_URL + img))) for img in TEST_IMGS])
    texts = tokenizer([template + l for l in LABELS], context_length=context_length)
    with torch.no_grad():
        image_features, text_features, logit_scale = model(images, texts)

        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        logits = logits.cpu().numpy()
        sorted_indices = sorted_indices.cpu().numpy()
    return sorted_indices, logits


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
        default=None,
        type=str,
        nargs="*",
        help='[Untested] Path to image as input. Can be a single string (eg: --image_path "URL1"), or a list of space-separated strings (eg: --image_path "URL1" "URL2")',
    )
    parser.add_argument(
        "--topk",
        default=1,
        type=int,
        help="topk num",
    )
    parser.add_argument(
        "--question",
        default="this is a photo of ",
        type=str,
        nargs="*",
        help='[Untested] question as input. Can be a single string (eg: --question "Q1"), or a list of space-separated strings (eg: --question "Q1" "Q2")',
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="[Untested] Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="[Untested] Whether to perform in bf16 precision.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="[Untested] Input batch size.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    args = parser.parse_args()

    #adapt_transformers_to_gaudi() fails on this
   

    precision ="fp32"
    if args.bf16:
        precision ="bf16"
        
    model, preprocess = create_model_from_pretrained(f"hf-hub:{args.model_name_or_path}", precision=precision)
    tokenizer = get_tokenizer(f"hf-hub:{args.model_name_or_path}")

    template = 'this is a photo of '

    device = torch.device('hpu') if torch.hpu.is_available() else torch.device('cpu')
    # warm up
    logger.info("Running warmup")
    for i in range(args.warmup):
        with torch.autocast(device_type="hpu", dtype=torch.float32, enabled=True):
            _, _ = run_qa(model, preprocess, tokenizer, template, device="hpu")
    logger.info("Running inference")
    start = time.time()
    for i in range(args.n_iterations):
        logits = None
        with torch.autocast(device_type="hpu", dtype=torch.float32, enabled=True):
            sorted_indices, logits = run_qa(model, preprocess, tokenizer, template, device="hpu")
        
        # Post Processing
        metadata_list = []
        for i, img in enumerate(TEST_IMGS):
            pred = LABELS[sorted_indices[i][0]]
            img_name = img.split('/')[-1]

            top_probs = []
            args.topk  = len(LABELS) if args.topk == -1 else args.topk
            for j in range(args.topk):
                jth_index = sorted_indices[i][j]
                top_probs.append(f"{LABELS[jth_index]}: {logits[i][jth_index] * 100:.1f}")

            metadata = {'filename': img_name, 'top_probs': '\n'.join(top_probs)}
            metadata_list.append(metadata)
    end = time.time()
    logger.info("Results:")
    pprint(metadata_list)
    logger.info(f"Inference Time per iteration = {(end-start) * 1000/args.n_iterations:.4}ms")
    logger.info(f"Throughput = {len(TEST_IMGS)*args.n_iterations/(end-start):.4} images/s")


if __name__ == "__main__":
    main()