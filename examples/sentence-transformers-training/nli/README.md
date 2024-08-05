# Natural Language Inference

Given two sentences (premise and hypothesis), the task of Natural Language Inference (NLI) is to decide if the premise entails the hypothesis, if they are contradiction, or if they are neutral. Commonly the NLI dataset in [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) are used.

The paper in [Conneau et al.](https://arxiv.org/abs/1705.02364) shows that NLI data can be quite useful when training Sentence Embedding methods. In [Sentence-BERT-Paper](https://arxiv.org/abs/1908.10084) NLI as a first fine-tuning step for sentence embedding methods has been used.

## Requirements

First, you should install the requirements:

```bash
pip install -U sentence-transformers
pip install git+https://github.com/huggingface/optimum-habana.git
```

## Single-card Training

To pre-train on the NLI task:

1. Choose a pre-trained model `<model_name>` (for example: [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)).

2. Load the training, validation, and test datasets. Below is an example of using the [AllNLI dataset](https://huggingface.co/datasets/sentence-transformers/all-nli) for training and validation, while the test set uses the STS Benchmark dataset.

```python
train_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train").select(range(10000))
eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="dev").select(range(1000))
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
```

3. Choose one of the following scripts based on the loss model:
   
	a. **[training_nli.py](training_nli.py)**:
	
	> This example uses `sentence_transformers.losses.SoftmaxLoss` as described in the original [Sentence Transformers paper](https://arxiv.org/abs/1908.10084).

	b. **[training_nli_v2.py](training_nli_v2.py)**:

	> The `sentence_transformers.losses.SoftmaxLoss` as used in our original SBERT paper does not yield optimal performance. A better loss is `sentence_transformers.losses.MultipleNegativesRankingLoss`, where we provide pairs or triplets. In this script, we provide a triplet of the format: (anchor, entailment_sentence, contradiction_sentence). The NLI data provides such triplets. The `sentence_transformers.losses.MultipleNegativesRankingLoss` yields much higher performances and is more intuitive than `sentence_transformers.losses.SoftmaxLoss`. We have used this loss to train the paraphrase model in our [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813) paper.

	c) **[training_nli_v3.py](training_nli_v3.py)**

	> Following the [GISTEmbed](https://arxiv.org/abs/2402.16829) paper, we can modify the in-batch negative selection from `sentence_transformers.losses.MultipleNegativesRankingLoss` using a guiding model. Candidate negative pairs are ignored during training if the guiding model considers the pair to be too similar. In practice, the `sentence_transformers.losses.GISTEmbedLoss` tends to produce a stronger training signal than `sentence_transformers.losses.MultipleNegativesRankingLoss` at the cost of some training overhead for running inference on the guiding model.

4. Execute the script:  

```bash
python training_nli.py bert-base-uncased
```

## Multi-card Training

For multi-card training you can use the script of [gaudi_spawn.py](https://github.com/huggingface/optimum-habana/blob/main/examples/gaudi_spawn.py) to execute. There are two options to run the multi-card training by using '--use_deepspeed' or '--use_mpi'. We take the option of '--use_deepspeed' for our example of  multi-card training. 

```bash
HABANA_VISIBLE_MODULES="2,3" python ../../gaudi_spawn.py --use_deepspeed --world_size 2 training_nli.py bert-base-uncased
```

## Dataset

We combine [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) into a dataset we call [AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli). These two datasets contain sentence pairs and one of three labels: entailment, neutral, contradiction:

| Sentence A (Premise)                                               | Sentence B (Hypothesis)                                            | Label         |
| ------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------- |
| A soccer game with multiple males playing.                         | Some men are playing a sport.                                      | entailment    |
| An older and younger man smiling.                                  | Two men are smiling and laughing at the cats playing on the floor. | neutral       |
| A man inspects the uniform of a figure in some East Asian country. | The man is sleeping.                                               | contradiction |

We format AllNLI in a few different subsets, compatible with different loss functions. See [triplet subset of AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/triplet) as example.

## SoftmaxLoss

<img src="https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SBERT_SoftmaxLoss.png" alt="SBERT SoftmaxLoss" width="250"/>

We pass the two sentences through our SentenceTransformer model and get the sentence embeddings _u_ and _v_. We then concatenate _u_, _v_ and _|u-v|_ to form one long vector. This vector is then passed to a softmax classifier, which predicts our three classes (entailment, neutral, contradiction).
