import csv
import gzip
import os
import time

import pytest
from sentence_transformers import SentenceTransformer, util

from .test_examples import TIME_PERF_FACTOR


if os.environ.get("GAUDI2_CI", "0") == "1":
    # Gaudi2 CI baselines
    MODELS_TO_TEST = [
        ("sentence-transformers/all-mpnet-base-v2", 762.5595168883357),
        ("sentence-transformers/multi-qa-mpnet-base-dot-v1", 545.3360251829846),
        ("sentence-transformers/all-distilroberta-v1", 958.5097903298335),
        ("sentence-transformers/all-MiniLM-L12-v2", 3614.2610109716247),
        ("sentence-transformers/multi-qa-distilbert-cos-v1", 944.6166139694299),
        ("sentence-transformers/all-MiniLM-L6-v2", 2615.6975354038477),
        ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 1208.3672807492396),
        ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 2392.1654748794062),
        ("sentence-transformers/paraphrase-albert-small-v2", 3896.1911011860166),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 3558.0778715789693),
        ("sentence-transformers/paraphrase-MiniLM-L3-v2", 5734.318427972881),
        ("sentence-transformers/distiluse-base-multilingual-cased-v1", 3487.3319366004903),
        ("sentence-transformers/distiluse-base-multilingual-cased-v2", 3807.2486282025716),
    ]
else:
    # Gaudi1 CI baselines
    MODELS_TO_TEST = [
        ("sentence-transformers/all-mpnet-base-v2", 164.36556936723508),
        ("sentence-transformers/multi-qa-mpnet-base-dot-v1", 116.82789535569364),
        ("sentence-transformers/all-distilroberta-v1", 226.90237421623164),
        ("sentence-transformers/all-MiniLM-L12-v2", 1252.6261862281467),
        ("sentence-transformers/multi-qa-distilbert-cos-v1", 216.47035182888888),
        ("sentence-transformers/all-MiniLM-L6-v2", 1109.160132821451),
        ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 471.14320842607674),
        ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 518.4762252952173),
        ("sentence-transformers/paraphrase-albert-small-v2", 1139.806075824319),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 1253.06776127632),
        ("sentence-transformers/paraphrase-MiniLM-L3-v2", 3029.398417051629),
        ("sentence-transformers/distiluse-base-multilingual-cased-v1", 947.844857744754),
        ("sentence-transformers/distiluse-base-multilingual-cased-v2", 947.7317550605878),
    ]


def _test_sentence_transformers(
    model_name: str,
    baseline: float,
):
    model = SentenceTransformer(model_name)

    nli_dataset_path = "/tmp/datasets/AllNLI.tsv.gz"
    sentences = set()
    max_sentences = 10000

    # Download datasets if needed
    if not os.path.exists(nli_dataset_path):
        util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

    with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            sentences.add(row["sentence1"])
            if len(sentences) >= max_sentences:
                break

    sentences = list(sentences)

    for i in range(2):
        start_time = time.perf_counter()
        _ = model.encode(sentences, batch_size=32)
        end_time = time.perf_counter()
        diff_time = end_time - start_time
        measured_throughput = len(sentences) / diff_time
    # Only assert the last measured throughtput as the first iteration is used as a warmup
    assert measured_throughput >= (2 - TIME_PERF_FACTOR) * baseline


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST)
def test_compute_embeddings_throughput(model_name: str, baseline: float):
    _test_sentence_transformers(model_name, baseline)
