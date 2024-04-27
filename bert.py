import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import wandb
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_metric

device = torch.device("cuda")
EPOCHS = 1
BATCH_SIZE = 10
LEARNING_RATE = 1e-5
SEED = 577


def dataset_conversion(train, test, val):

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)
    val_dataset = Dataset.from_pandas(val)

    return DatasetDict({"train": train_dataset,
                        "test": test_dataset,
                        "val": val_dataset})


def tokenize_function(dataset):
    return tokenizer(dataset["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    metric_acc = load_metric("accuracy")
    metric_rec = load_metric("recall")
    metric_pre = load_metric("precision")
    metric_f1 = load_metric("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    recall = metric_rec.compute(predictions=predictions, references=labels, average="micro")["recall"]
    precision = metric_pre.compute(predictions=predictions, references=labels, average="micro")["precision"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="micro")["f1"]

    return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}


data = pd.read_csv('Suicide_Detection_cleaned-2.csv', header=0)
data.reset_index(drop=True, inplace=True)
data.replace({"class": {"SuicideWatch": 2, "depression": 1, "teenagers": 0}}, inplace=True)
data.drop(columns=['text'], inplace=True)
data = data.rename(columns={"cleaned_text": "text"})
data.head(10)
data = data.rename(columns={"class": "label"})

train, other = train_test_split(data, random_state=SEED, test_size=0.2, stratify=data['label'])
test, val = train_test_split(other, random_state=SEED, test_size=0.5, stratify=other['label'])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
raw_datasets = dataset_conversion(train, test, val)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]
val_dataset = tokenized_datasets["val"]

# Import BERT-base pretrained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

WANDB_ENTITY = "mahlawat"
WANDB_PROJECT = "cs577"
WANDB_RUN = "bert"
wandb.login()
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN)

# Define model and training parameters
training_args = TrainingArguments(output_dir="Models/bert_checkpoint", overwrite_output_dir = True, report_to = 'wandb',
    learning_rate=LEARNING_RATE, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE, seed=SEED, run_name=WANDB_RUN, logging_dir="Models/bert_checkpoint/logs",
    save_strategy="steps", save_steps=1500)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
    tokenizer=tokenizer, compute_metrics=compute_metrics)

trainer.train()
wandb.finish()

trainer.save_model("Models/bert")

trainer.evaluate()
trainer.predict(test_dataset).metrics