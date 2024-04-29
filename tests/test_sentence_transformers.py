import json
import gzip
import csv
import os
import re
import time
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from sentence_transformers import SentenceTransformer, util

import pytest

from .test_examples import TIME_PERF_FACTOR


if os.environ.get("GAUDI2_CI", "0") == "1":
    # Gaudi2 CI baselines
    MODELS_TO_TEST = [
        ("sentence-transformers/all-mpnet-base-v2", 762.5595168883357),
    ]
else:
    # Gaudi1 CI baselines
    MODELS_TO_TEST = [
        ("sentence-transformers/all-mpnet-base-v2", 0.0),
    ]


def _test_sentence_transformers(
    model_name: str,
    baseline: float,
):
    model = SentenceTransformer(model_name)

    nli_dataset_path = "datasets/AllNLI.tsv.gz"
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
        emb = model.encode(sentences, batch_size=32)
        end_time = time.perf_counter()
        diff_time = end_time - start_time
        measured_throughput = len(sentences) / diff_time
    # Only assert the last measured throughtput as the first iteration is used as a warmup
    assert measured_throughput >= (2 - TIME_PERF_FACTOR) * baseline


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST)
def test_compute_embeddings_throughput(model_name: str, baseline: float):
    _test_sentence_transformers(model_name, baseline)
