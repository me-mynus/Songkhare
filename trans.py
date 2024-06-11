# Transformers -> encoders and decoders
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score

from transformers import Trainer

tqdm.pandas()

classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]
train = pd.read_json("data/transform/train.jsonl", lines=True)
valid = pd.read_json("data/transform/validation.jsonl", lines=True)

# lambda function maps the label indexes with the classes array
train["label_name"] = train["label"].apply(lambda x: classes[x])
valid["label_name"] = valid["label"].apply(lambda x: classes[x])

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize(text):
    val = tokenizer(text, padding=True, truncation=True)
    return val


tokenized = train["text"].progress_apply(tokenize)
tokenized_val = valid["text"].progress_apply(tokenize)

train["input_ids"] = tokenized.apply(lambda x: x["input_ids"])
train["attention_mask"] = tokenized.apply(lambda x: x["attention_mask"])

valid["input_ids"] = tokenized_val.apply(lambda x: x["input_ids"])
valid["attention_mask"] = tokenized_val.apply(lambda x: x["attention_mask"])


inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
model = AutoModel.from_pretrained("distilbert-base-uncased")

with torch.no_grad():
    outputs = model(**inputs)


last_hidden_states = outputs.last_hidden_state

num_labels = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels
).to(device)


batch_size = 64
model_name = "distilbert-emotion"

training_args = TrainingArguments(
    output_dir=f"model_data/{model_name}",
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    disable_tqdm=False,
    weight_decay=0.01,
    eval_strategy="epoch",
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=valid,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()
