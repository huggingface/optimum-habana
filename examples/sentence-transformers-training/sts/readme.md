# Semantic Textual Similarity

Semantic Textual Similarity (STS) assigns a score on the similarity of two texts. In this example, we use the [stsb](https://huggingface.co/datasets/sentence-transformers/stsb) dataset as training data to fine-tune our model. See the following example scripts how to tune SentenceTransformer on STS data:

- **[training_stsbenchmark.py](training_stsbenchmark.py)** - This example shows how to create a SentenceTransformer model from scratch by using a pre-trained transformer model (e.g. [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)) together with a pooling layer.
- **[training_stsbenchmark_continue_training.py](training_stsbenchmark_continue_training.py)** - This example shows how to continue training on STS data for a previously created & trained SentenceTransformer model (e.g. [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)).

## Requirements

First, you should install the requirements:
```bash
pip install -U sentence-transformers
pip install git+https://github.com/huggingface/optimum-habana.git
```


## Usage

To fine training on STS you can do following steps -

1) choose the pretrained model as model_name command line below as args
You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base

2) Choose the training dataset like here we used 'sentence-transformers/stsb'.
```bash
    train_dataset = load_dataset("sentence-transformers/stsb", split="train")
    eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
```

3) Choose the test dataset
```bash
    test_dataset = load_dataset("sentence-transformers/stsb", split="test")
```

4) define the loss_model, TrainingArguments and Trainer

5) execute the training command in Single-Card

```bash
python examples/sentence-transformers-training/sts/training_stsbenchmark.py model_name
```

6) execute the training command in Multi-Card (2 cards, hpu2/3)

```bash
HABANA_VISIBLE_MODULES="2,3" python ./gaudi_spawn.py --use_deepspeed --world_size 2 sentence-transformers-training/sts/training_stsbenchmark.py model_name
```

## Training data
```eval_rst
In STS, we have sentence pairs annotated together with a score indicating the similarity. In the original STSbenchmark dataset, the scores range from 0 to 5. We have normalized these scores to range between 0 and 1 in `stsb <https://huggingface.co/datasets/sentence-transformers/stsb>`_, as that is required for :class:`~sentence_transformers.losses.CosineSimilarityLoss`.
```

Here is a simplified version of our training data:

```python
from datasets import Dataset

sentence1_list = ["My first sentence", "Another pair"]
sentence2_list = ["My second sentence", "Unrelated sentence"]
labels_list = [0.8, 0.3]
train_dataset = Dataset.from_dict({
    "sentence1": sentence1_list,
    "sentence2": sentence2_list,
    "label": labels_list,
})
# => Dataset({
#     features: ['sentence1', 'sentence2', 'label'],
#     num_rows: 2
# })
print(train_dataset[0])
# => {'sentence1': 'My first sentence', 'sentence2': 'My second sentence', 'label': 0.8}
print(train_dataset[1])
# => {'sentence1': 'Another pair', 'sentence2': 'Unrelated sentence', 'label': 0.3}
```

In the aforementioned scripts, we directly load the [stsb](https://huggingface.co/datasets/sentence-transformers/stsb) dataset:

```python
from datasets import load_dataset

train_dataset = load_dataset("sentence-transformers/stsb", split="train")
# => Dataset({
#     features: ['sentence1', 'sentence2', 'score'],
#     num_rows: 5749
# })
```

## Loss Function
```eval_rst
We use :class:`~sentence_transformers.losses.CosineSimilarityLoss` as our loss function.
```

<img src="https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SBERT_Siamese_Network.png" alt="SBERT Siamese Network Architecture" width="250"/>

For each sentence pair, we pass sentence A and sentence B through the BERT-based model, which yields the embeddings *u* und *v*. The similarity of these embeddings is computed using cosine similarity and the result is compared to the gold similarity score. Note that the two sentences are fed through the same model rather than two separate models. In particular, the cosine similarity for similar texts is maximized and the cosine similarity for dissimilar texts is minimized. This allows our model to be fine-tuned and to recognize the similarity of sentences.

For more details, see [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084).

```
