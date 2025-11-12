import json
import os
import sys
import random
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import mean_squared_error
import logging
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/res_scorer"
DATASET_PATH = "train-data/res_bench_2k.json"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_res_model.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_or_create_dataset():
    if os.path.exists(DATASET_PATH):
        logger.info(f"Loading RES dataset from {DATASET_PATH}")
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and "response" in data[0]:
            data = [{"text": d["response"], "score": d["explainability_score"]} for d in data]
    else:
        print("Data not present")
        data = []
    return data

class DistilBertForRegression(torch.nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.distilbert.config.dim, 1)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.regressor(outputs.last_hidden_state[:, 0]).squeeze(-1)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(logits, labels.float())
        return {"loss": loss, "logits": logits}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    rmse = np.sqrt(mean_squared_error(labels, logits))
    mae = np.mean(np.abs(labels - logits))
    accuracy_within_0_5 = np.mean(np.abs(labels - logits) <= 0.5)
    return {"rmse": rmse, "mae": mae, "accuracy_within_0_5": accuracy_within_0_5}

class RES_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs["loss"]
        if loss is None and labels is not None:
            logits = outputs["logits"]
            loss = torch.nn.functional.mse_loss(logits, labels.float())
            outputs["loss"] = loss
        if return_outputs:
            return (loss, outputs)
        return loss

def main():
    logger.info("Starting RES model training pipeline")
    raw_data = load_or_create_dataset()
    if not raw_data:
        logger.error("Dataset could not be loaded. Exiting.")
        return

    dataset = Dataset.from_list([{"text": d["text"], "labels": float(d["score"])} for d in raw_data])
    dataset = dataset.train_test_split(test_size=0.2)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    tokenized = dataset.map(
        lambda x: {
            **tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"),
            "labels": x["labels"]
        },
        batched=True,
        remove_columns=["text"]
    )

    model = DistilBertForRegression(MODEL_NAME)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs/res",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        report_to="none"
    )

    trainer = RES_Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logger.info("Beginning training...")
    trainer.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/pytorch_model.bin")
    tokenizer.save_pretrained(OUTPUT_DIR)

    config = {
        "model_type": "distilbert-res-regression",
        "min_score": 1.0,
        "max_score": 3.0,
        "version": "1.0",
        "training_samples": len(raw_data)
    }
    with open(f"{OUTPUT_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"RES model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
