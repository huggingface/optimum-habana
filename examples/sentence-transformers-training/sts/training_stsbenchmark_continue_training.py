"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from Hugging Face.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""

import logging
import sys
import traceback
from datetime import datetime

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments, SentenceTransformerGaudiTrainer
from optimum.habana import SentenceTransformerGaudiTrainingArguments

from optimum.habana.sentence_transformers.modeling_utils import adapt_sentence_transformers_to_gaudi
adapt_sentence_transformers_to_gaudi()


def main():

    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    
    # You can specify any Sentence Transformer model here, for example all-mpnet-base-v2, all-MiniLM-L6-v2, mixedbread-ai/mxbai-embed-large-v1
    model_name = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-mpnet-base-v2"
    train_batch_size = 16
    num_epochs = 4
    output_dir = (
        "output/training_stsbenchmark_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    
    # 1. Here we define our SentenceTransformer model.
    model = SentenceTransformer(model_name)
    
    # 2. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
    train_dataset = load_dataset("sentence-transformers/stsb", split="train")
    eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
    test_dataset = load_dataset("sentence-transformers/stsb", split="test")
    logging.info(train_dataset)
    
    # 3. Define our training loss
    # CosineSimilarityLoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) needs two text columns and one
    # similarity score column (between 0 and 1)
    train_loss = losses.CosineSimilarityLoss(model=model)
    # train_loss = losses.CoSENTLoss(model=model)
    
    # 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["sentence1"],
        sentences2=eval_dataset["sentence2"],
        scores=eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )
    
    # 5. Define the training arguments
    args = SentenceTransformerGaudiTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        warmup_ratio=0.1,
        #fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        #bf16=False,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="sts",  # Will be used in W&B if `wandb` is installed
        use_habana = True,
        gaudi_config_name = 'Habana/distilbert-base-uncased',
        use_lazy_mode = True,
        use_hpu_graphs = True,
        use_hpu_graphs_for_inference = False,
        use_hpu_graphs_for_training = True,
    
    )
    
    # 6. Create the trainer & start training
    trainer = SentenceTransformerGaudiTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    trainer.train()
    
    
    # 7. Evaluate the model performance on the STS Benchmark test dataset
    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=test_dataset["sentence1"],
        sentences2=test_dataset["sentence2"],
        scores=test_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-test",
    )
    test_evaluator(model)
    
    # 8. Save the trained & evaluated model locally
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)

if __name__ == "__main__":
    main()

