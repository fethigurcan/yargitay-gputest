import numpy as np
import evaluate
import mlflow

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    # import dataset
    dataset = load_dataset("yelp_review_full")
    
    # import tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    # tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # shuffle and limit the dataset
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(5000))
    
    # import training metric
    metric = evaluate.load("accuracy")

    # create a trainer
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", report_to="none")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    mlflow.end_run()

    