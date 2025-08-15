"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

"""

import logging
from datetime import datetime

import pytest
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction

from optimum.habana import SentenceTransformerGaudiTrainer, SentenceTransformerGaudiTrainingArguments
from optimum.habana.sentence_transformers.modeling_utils import adapt_sentence_transformers_to_gaudi


adapt_sentence_transformers_to_gaudi()


@pytest.mark.parametrize("peft", [False, True])
def test_training_stsbenchmark(peft):
    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    # You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    model_name = "distilbert-base-uncased"
    train_batch_size = 16
    num_epochs = 1
    output_dir = (
        "output/training_stsbenchmark_"
        + model_name.replace("/", "-")
        + "-"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    # 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
    # create one with "mean" pooling.
    model = SentenceTransformer(model_name)
    if peft:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=16,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            target_modules=["q_lin", "k_lin", "v_lin"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 2. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
    train_dataset = load_dataset("sentence-transformers/stsb", split="train")
    eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
    test_dataset = load_dataset("sentence-transformers/stsb", split="test")
    logging.info(train_dataset)

    # 3. Define our training loss
    # CosineSimilarityLoss (https://sbert.net/docs/package_reference/losses.html#cosentloss) needs two text columns and one
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
    # args = SentenceTransformerTrainingArguments()
    ### args = SentenceTransformerTrainingArguments(
    args = SentenceTransformerGaudiTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        warmup_ratio=0.1,
        # fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        # bf16=False,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="sts",  # Will be used in W&B if `wandb` is installed
        use_habana=True,
        gaudi_config_name="Habana/distilbert-base-uncased",
        use_lazy_mode=True,
        use_hpu_graphs=True,
        use_hpu_graphs_for_inference=False,
        use_hpu_graphs_for_training=True,
    )

    # 6. Create the trainer & start training
    # trainer = SentenceTransformerTrainer(
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
    # final_output_dir = f"{output_dir}/final"
    # model.save(final_output_dir)
