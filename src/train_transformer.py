"""Fine-tune a multilingual Transformer for French sentiment analysis."""

import json
import inspect

import evaluate
import numpy as np
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import (
    BATCH_SIZE,
    LABEL_COLUMN,
    LEARNING_RATE,
    MAX_LENGTH,
    NUM_EPOCHS,
    OUTPUTS_DIR,
    TRANSFORMER_EVAL_SAMPLE_SIZE,
    TRANSFORMER_MODEL_DIR,
    TRANSFORMER_MODEL_NAME,
    TRANSFORMER_TRAIN_SAMPLE_SIZE,
)
from data_utils import clean_dataset_split, load_sentiment_dataset, sample_split


accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_metric.compute(
            predictions=predictions, references=labels, average="weighted", zero_division=0
        )["precision"],
        "recall": recall_metric.compute(
            predictions=predictions, references=labels, average="weighted", zero_division=0
        )["recall"],
        "f1": f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["f1"],
    }


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_sentiment_dataset()
    train_dataset = clean_dataset_split(sample_split(dataset["train"], TRANSFORMER_TRAIN_SAMPLE_SIZE))
    eval_dataset = clean_dataset_split(sample_split(dataset["validation"], TRANSFORMER_EVAL_SAMPLE_SIZE))

    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["review"], truncation=True, max_length=MAX_LENGTH)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column(LABEL_COLUMN, "labels")
    eval_dataset = eval_dataset.rename_column(LABEL_COLUMN, "labels")

    model_config = BertConfig.from_pretrained(TRANSFORMER_MODEL_NAME, num_labels=2)
    model = BertForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_NAME,
        config=model_config,
    )

    training_args_kwargs = {
        "output_dir": str(OUTPUTS_DIR / "transformer_training"),
        "save_strategy": "epoch",
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "num_train_epochs": NUM_EPOCHS,
        "weight_decay": 0.01,
        "load_best_model_at_end": False,
        "save_total_limit": 1,
        "logging_steps": 50,
        "report_to": "none",
        "seed": 42,
    }
    training_args_signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_signature:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("Fine-tuning du Transformer...")
    trainer.train()

    print("Evaluation finale...")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    trainer.save_model(str(TRANSFORMER_MODEL_DIR))
    tokenizer.save_pretrained(str(TRANSFORMER_MODEL_DIR))

    with open(OUTPUTS_DIR / "transformer_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nModele sauvegarde dans: {TRANSFORMER_MODEL_DIR}")


if __name__ == "__main__":
    main()
