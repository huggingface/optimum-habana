# Examples for Sentence Transformers

We provide 3 examples to show how to use the Sentence Transformers with HPU devices. 

- **[training_stsbenchmark.py](https://github.com/huggingface/optimum-habana/tree/main/examples/sentence-transformers-training/sts)** - This example shows how to create a SentenceTransformer model from scratch by using a pre-trained transformer model (e.g. [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)) together with a pooling layer.

- **[training_nli.py](https://github.com/huggingface/optimum-habana/tree/main/examples/sentence-transformers-training/nli)** - This example provides two sentences (a premise and a hypothesis), and the task of Natural Language Inference (NLI) is to determine whether the premise entails the hypothesis, contradicts it, or if they are neutral. Commonly the NLI dataset in [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) are used.

- **[training_paraphrases.py](https://github.com/huggingface/optimum-habana/tree/main/examples/sentence-transformers-training/paraphrases)** - This example loads various datasets from the Sentence Transformers. We construct batches by sampling examples from the respective dataset. 

### Tested Examples/Models and Configurations

The following table contains examples supported and configurations we have validated on Gaudi2. 

| Examples                    |  General  | e5-mistral-7b-instruct | BF16 | Single Card | Multi-Cards |
|-----------------------------|-----------|------------|------|-------------|-------------|
| training_nli.py             |     ✔     |      ✔    |   ✔  |     ✔       |     ✔      |
| training_stsbenchmark.py    |     ✔     |      ✔    |   ✔  |     ✔       |     ✔      |
| training_paraphrases.py     |     ✔     |           |       |     ✔       |            |

Notice: 
1. In the table, the column 'General' refers to general models like mpnet, MiniLM.
2. When e5-mistral-7b-instruct model is enabled for the test, single card will use the LoRA + gradient_checkpoint and multi-card will use the deepspeed zero2/zero3 stage to reduce the memory requirement. 
3. For the detailed instructions on how to run each example, you can refer to the README file located in each example folder. 
