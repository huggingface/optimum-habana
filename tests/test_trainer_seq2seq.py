# coding=utf-8
# Copyright 2022 the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration
from transformers.testing_utils import TestCasePlus, require_torch
from transformers.utils import is_datasets_available

from optimum.habana import GaudiConfig, GaudiSeq2SeqTrainer, GaudiSeq2SeqTrainingArguments
from optimum.habana.transformers.generation import GaudiGenerationConfig
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


if is_datasets_available():
    import datasets


adapt_transformers_to_gaudi()


class GaudiSeq2seqTrainerTester(TestCasePlus):
    @require_torch
    def test_finetune_t5(self):
        train_dataset = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:1%]")
        val_dataset = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation[:1%]")

        train_dataset = train_dataset.select(range(32))
        val_dataset = val_dataset.select(range(16))

        batch_size = 4

        training_args = GaudiSeq2SeqTrainingArguments(
            output_dir=self.get_auto_remove_tmp_dir(),
            gaudi_config_name="Habana/t5",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            do_train=True,
            do_eval=True,
            use_habana=True,
            use_lazy_mode=True,
            use_hpu_graphs_for_inference=True,
            report_to="none",
        )

        model = T5ForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-t5-v1.1")
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

        model.config.max_length = 128

        def _map_to_encoder_decoder_inputs(batch):
            # Tokenizer will automatically set [BOS] <text> [EOS]
            inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512)
            outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=128)
            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask

            batch["decoder_input_ids"] = outputs.input_ids
            batch["labels"] = outputs.input_ids.copy()
            batch["decoder_attention_mask"] = outputs.attention_mask

            assert all(len(x) == 512 for x in inputs.input_ids)
            assert all(len(x) == 128 for x in outputs.input_ids)

            return batch

        def _compute_metrics(pred):
            labels_ids = pred.label_ids
            pred_ids = pred.predictions

            # all unnecessary tokens are removed
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

            accuracy = sum([int(pred_str[i] == label_str[i]) for i in range(len(pred_str))]) / len(pred_str)

            return {"accuracy": accuracy}

        # map train dataset
        train_dataset = train_dataset.map(
            _map_to_encoder_decoder_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "highlights"],
        )
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"],
        )

        # same for validation dataset
        val_dataset = val_dataset.map(
            _map_to_encoder_decoder_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "highlights"],
        )
        val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"],
        )

        # instantiate trainer
        trainer = GaudiSeq2SeqTrainer(
            model=model,
            args=training_args,
            compute_metrics=_compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )

        # start training
        trainer.train()

        # start evaluation using greedy search
        trainer.evaluate(max_length=model.config.max_length, num_beams=1)

        # start evaluation using beam search
        trainer.evaluate(max_length=model.config.max_length, num_beams=2)

    @require_torch
    def test_bad_generation_config_fail_early(self):
        # Tests that a bad generation config causes the trainer to fail early
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt", padding="longest")
        gen_config = GaudiGenerationConfig(
            do_sample=False, top_p=0.9
        )  # bad: top_p is not compatible with do_sample=False

        training_args = GaudiSeq2SeqTrainingArguments(
            output_dir="tmp_trainer",
            predict_with_generate=True,
            generation_config=gen_config,
            use_habana=True,
            use_lazy_mode=True,
            report_to="none",
        )
        with self.assertRaises(ValueError) as exc:
            _ = GaudiSeq2SeqTrainer(
                model=model,
                gaudi_config=GaudiConfig(),
                args=training_args,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=lambda x: {"samples": x[0].shape[0]},
            )
        self.assertIn("The loaded generation config instance is invalid", str(exc.exception))
