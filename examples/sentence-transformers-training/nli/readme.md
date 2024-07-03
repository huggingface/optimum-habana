# Natural Language Inference

Given two sentence (premise and hypothesis), Natural Language Inference (NLI) is the task of deciding if the premise entails the hypothesis, if they are contradiction, or if they are neutral. Commonly used NLI dataset are [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli). 

[Conneau et al.](https://arxiv.org/abs/1705.02364) showed that NLI data can be quite useful when training Sentence Embedding methods. We also found this in our [Sentence-BERT-Paper](https://arxiv.org/abs/1908.10084) and often use NLI as a first fine-tuning step for sentence embedding methods.


## Requirements

First, you should install the requirements:
```bash
pip install -U sentence-transformers
pip install git+https://github.com/huggingface/optimum-habana.git
```


## Usage

To training on NLI -

```bash
python examples/sentence-transformers-training/nli/training_nli.py model_name
```



## Data
We combine [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) into a dataset we call [AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli). These two datasets contain sentence pairs and one of three labels: entailment, neutral, contradiction:

| Sentence A (Premise) | Sentence B (Hypothesis) | Label |
| --- | --- | --- |
| A soccer game with multiple males playing. | Some men are playing a sport. | entailment |
| An older and younger man smiling. | Two men are smiling and laughing at the cats playing on the floor. | neutral |
| A man inspects the uniform of a figure in some East Asian country. | The man is sleeping. | contradiction |

We format AllNLI in a few different subsets, compatible with different loss functions. See for example the [triplet subset of AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/triplet).

## SoftmaxLoss
```eval_rst
`Conneau et al. <https://arxiv.org/abs/1705.02364>`_ described how a softmax classifier on top of a `siamese network <https://en.wikipedia.org/wiki/Siamese_neural_network>`_ can be used to learn meaningful sentence representation. We can achieve this by using :class:`~sentence_transformers.losses.SoftmaxLoss`:
```

<img src="https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SBERT_SoftmaxLoss.png" alt="SBERT SoftmaxLoss" width="250"/>

We pass the two sentences through our SentenceTransformer model and get the sentence embeddings *u* and *v*. We then concatenate *u*, *v* and *|u-v|* to form one long vector. This vector is then passed to a softmax classifier, which predicts our three classes (entailment, neutral, contradiction).

