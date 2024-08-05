# Paraphrases Training

## Requirements

First, you should install the requirements:

```bash
pip install -U sentence-transformers
pip install git+https://github.com/huggingface/optimum-habana.git
```

## Usage

To fine-tune on the paraphrase task:

1. Choose a pre-trained model `<model_name>` (For example: `bert-base-uncased`).

2. Choose the training, evaluation, and test dataset(s). Here, we use a dataset dictionary to include multiple datasets.

```python
all_nli_train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
sentence_compression_train_dataset = load_dataset("sentence-transformers/sentence-compression", split="train")
simple_wiki_train_dataset = load_dataset("sentence-transformers/simple-wiki", split="train")
altlex_train_dataset = load_dataset("sentence-transformers/altlex", split="train")
quora_train_dataset = load_dataset("sentence-transformers/quora-duplicates", "triplet", split="train")
coco_train_dataset = load_dataset("sentence-transformers/coco-captions", split="train")
flickr_train_dataset = load_dataset("sentence-transformers/flickr30k-captions", split="train")
yahoo_answers_train_dataset = load_dataset(
    "sentence-transformers/yahoo-answers", "title-question-answer-pair", split="train"
)
stack_exchange_train_dataset = load_dataset(
    "sentence-transformers/stackexchange-duplicates", "title-title-pair", split="train"
)

train_dataset_dict = {
    "all-nli": all_nli_train_dataset,
    "sentence-compression": sentence_compression_train_dataset,
    "simple-wiki": simple_wiki_train_dataset,
    "altlex": altlex_train_dataset,
    "quora-duplicates": quora_train_dataset,
    "coco-captions": coco_train_dataset,
    "flickr30k-captions": flickr_train_dataset,
    "yahoo-answers": yahoo_answers_train_dataset,
    "stack-exchange": stack_exchange_train_dataset,
}
# Eval dataset
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
# Test dataset
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
```

3. Run the training command:

```bash
python training_paraphrases.py distilroberta-base
```

## Paraphrase Dataset

The [training_paraphrases.py](training_paraphrases.py) script loads various datasets from the sentence transformers. We construct batches by sampling examples from the respective dataset. So far, examples are not mixed between the datasets, i.e., a batch consists only of examples from a single dataset.

As the dataset sizes are quite different in size, we perform round-robin sampling from sentence transformers to train using the same amount of batches from each dataset.

## Pre-Trained Models

Have a look at [pre-trained models](https://github.com/UKPLab/sentence-transformers/blob/master/docs/sentence_transformer/pretrained_models.md) to view all models that were trained on these paraphrase datasets.

- [paraphrase-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L12-v2) - Trained on the following datasets: AllNLI, sentence-compression, SimpleWiki, altlex, msmarco-triplets, quora_duplicates, coco_captions,flickr30k_captions, yahoo_answers_title_question, S2ORC_citation_pairs, stackexchange_duplicate_questions, wiki-atomic-edits
- [paraphrase-distilroberta-base-v2](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v2) - Trained on the following datasets: AllNLI, sentence-compression, SimpleWiki, altlex, msmarco-triplets, quora_duplicates, coco_captions,flickr30k_captions, yahoo_answers_title_question, S2ORC_citation_pairs, stackexchange_duplicate_questions, wiki-atomic-edits
- [paraphrase-distilroberta-base-v1](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1) - Trained on the following datasets: AllNLI, sentence-compression, SimpleWiki, altlex, quora_duplicates, wiki-atomic-edits, wiki-split
- [paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1) - Multilingual version of paraphrase-distilroberta-base-v1, trained on parallel data for 50+ languages. (Teacher: [paraphrase-distilroberta-base-v1](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1), Student: [xlm-r-base](https://huggingface.co/FacebookAI/xlm-roberta-base))
