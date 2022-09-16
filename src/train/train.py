"""
This script contains the code for finetuning a pretrained mT5 model for summarisation
on the DaNewsroom dataset.

This code can be run using

```bash
python train.py
```

if you want to overwrite specific parameters, you can do so by passing them as arguments to the script, e.g.

```bash
python train.py --config_file config.yaml training_data.max_input_length=512
```

or by passing a config file with the `--config_file` flag, e.g.

```bash
python train.py --config_file config.yaml
```
"""

import time
from functools import partial

import datasets
import hydra
import nltk
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from OmegaConf import DictConfig
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
)

from .utils import flatten_nested_config


def load_dataset(cfg) -> Dataset:
    """
    Load the cleaned danewsroom dataset
    """
    cfg = cfg.training_data
    # Load data
    train = Dataset.from_pandas(
        pd.read_csv(
            cfg.dataset.train_path,
            usecols=[cfg.text_column, cfg.summary_column],
        )
    )
    val = Dataset.from_pandas(
        pd.read_csv(
            cfg.dataset.val_path,
            usecols=[cfg.text_column, cfg.summary_column],
        )
    )
    test = Dataset.from_pandas(
        pd.read_csv(
            cfg.dataset.test_path,
            usecols=[cfg.text_column, cfg.summary_column],
        )
    )
    # make into datasetdict format
    return datasets.DatasetDict({"train": train, "validation": val, "test": test})


def preprocess_function(examples, tokenizer, cfg):
    cfg = cfg.training_data
    inputs = [doc for doc in examples["text"]]

    # tokenize the input + truncate to max input length
    model_inputs = tokenizer(inputs, max_length=cfg.max_input_length, truncation=True)

    # tokenize the ref summary + truncate to max input length
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"], max_length=cfg.max_target_length, truncation=True
        )

    # getting IDs for token
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def generate_summary(batch, tokenizer, model, cfg):
    # prepare test data
    batch["text"] = [doc for doc in batch["text"]]
    inputs = tokenizer(
        batch["text"],
        padding="max_length",
        return_tensors="pt",
        max_length=cfg.training.max_input_length,
        truncation=True,
    )
    input_ids = inputs.input_ids.to(cfg.device)
    attention_mask = inputs.attention_mask.to(cfg.device)

    # make the model generate predictions (summaries) for articles in text set
    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


def compute_metrics(eval_pred, tokenizer, cfg):

    rouge_metric = datasets.load_metric("rouge")
    bert_metric = datasets.load_metric("bertscore")

    predictions, labels = eval_pred  # labels = the reference summaries
    # decode generated summaries from IDs to actual words
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # replace -100 in the labels as we can't decode them, replace with pad token id instead
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decode reference summaries from IDs to actual words
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    # compute ROUGE scores
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {key: value.mid.fmeasure for key, value in result.items()}

    # compute BERTScores
    bertscores = bert_metric.compute(
        predictions=decoded_preds, references=decoded_labels, lang=cfg.language
    )
    result["bertscore"] = np.mean(bertscores["precision"])

    # add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    # round to 4 decimals
    metrics = {k: round(v, 4) for k, v in result.items()}

    return metrics


@hydra.main(config_path="configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    """
    Main function for training the model.
    """
    # Setup
    # Setting up wandb
    wandb.init(project="da-newsroom-summerization", config=flatten_nested_config(cfg), mode=cfg.wandb_mode)
    nltk.download("punkt")

    # load dataset
    dataset = load_dataset(cfg)
    start = time.time()

    # Preprocessing
    # removed fast because of warning message
    tokenizer = T5Tokenizer.from_pretrained(cfg.model_checkpoint)

    # make the tokenized datasets using the preprocess function
    _preprocess = partial(preprocess_function, tokenizer=tokenizer, cfg=cfg)
    tokenized_datasets = dataset.map(_preprocess, batched=True)

    # Fine-tuning
    # load the pretrained mT5 model from the Huggingface hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_checkpoint,
        min_length=cfg.model.min_length,
        max_length=cfg.model.max_length,
        num_beams=cfg.model.num_beams,
        no_repeat_ngram_size=cfg.model.no_repeat_ngram_size,
        length_penalty=cfg.model.length_penalty,
        early_stopping=cfg.model.early_stopping,
        dropout_rate=cfg.model.dropout_rate,
    )

    # specify training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir + wandb.run.name,
        evaluation_strategy=cfg.training.evaluation_strategy,
        save_strategy=cfg.training.save_strategy,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        warmup_steps=cfg.training.warmup_steps,
        save_total_limit=cfg.training.save_total_limit,
        num_train_epochs=cfg.training.num_train_epochs,
        predict_with_generate=cfg.training.predict_with_generate,
        overwrite_output_dir=cfg.training.overwrite_output_dir,
        fp16=cfg.training.fp16,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
    )

    # pad the articles and ref summaries (with -100) to max input length
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, pad_to_multiple_of=cfg.training.pad_to_multiple_of
    )

    # make the model trainer
    _compute_metrics = partial(compute_metrics, tokenizer=tokenizer, cfg=cfg)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    # train the model!
    trainer.train()

    # Testing
    model.to(cfg.device)
    test_data = dataset["test"]

    # generate summaries for test set with the function
    _generate_summary = partial(generate_summary, model=model, tokenizer=tokenizer)
    results = test_data.map(
        _generate_summary,
        batched=True,
        batch_size=cfg.training.per_device_eval_batch_size,
    )

    pred_str = results["pred"]  # the model's generated summary
    label_str = results["summary"]  # actual ref summary from test set

    # compute rouge scores
    rouge_metric = datasets.load_metric("rouge")
    bert_metric = datasets.load_metric("bertscore")
    rouge_output = rouge_metric.compute(predictions=pred_str, references=label_str)
    bert_output = bert_metric.compute(
        predictions=pred_str, references=label_str, lang=cfg.language
    )

    # save predictions and rouge scores on test set
    results = pd.DataFrame([results])
    results.to_csv(cfg.training.output_dir + "/" + wandb.run.name + "_preds.csv")

    rouge_output = pd.DataFrame([rouge_output])
    rouge_output.to_csv(cfg.training.output_dir + "/" + wandb.run.name + "_rouge.csv")

    bert_output = pd.DataFrame([bert_output])
    bert_output.to_csv(cfg.training.output_dir + "/" + wandb.run.name + "_bert.csv")

    end = time.time()
    print("TIME SPENT:")
    print(end - start)


if __name__ == "__main__":
    main()
