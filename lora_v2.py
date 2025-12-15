# =====================
# LoRA POS SWEEP (CLUSTER + WANDB)
# =====================

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, confusion_matrix

from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
    Trainer,
    TrainingArguments
)

from peft import LoraConfig, TaskType, get_peft_model

from tokenizer import token_label_alignment
from data_utils import load_pickle
import wandb


# ---------------------
# PATHS (CLUSTER SAFE)
# ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "Clean_Datasets",
    "cleaned_ud_dataset(cat)_prueba1000.pkl"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "pos_tagger_lora_sweep")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 128
SMOKE_TEST = False


# ---------------------
# DEVICE
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------
# LOAD DATA
# ---------------------
dataset = load_pickle(DATA_PATH)
if SMOKE_TEST:
    dataset = dataset[:64]


# ---------------------
# TOKENIZER
# ---------------------
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")


# ---------------------
# LABEL VOCAB
# ---------------------
all_tags = set()
for _, tags in dataset:
    for tag in tags:
        all_tags.add(tag if tag is not None else "X")

UPOS = sorted(all_tags)
upos2id = {t: i for i, t in enumerate(UPOS)}
id2upos = {i: t for t, i in upos2id.items()}
num_labels = len(UPOS)


# ---------------------
# DATASET
# ---------------------
class POSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tags = self.data[idx]
        encoding, aligned = token_label_alignment(tokens, tags, tokenizer)

        labels = [
            -100 if l in (-100, None) else upos2id.get(l, -100)
            for l in aligned
        ]

        return {
            "input_ids": encoding["input_ids"][:MAX_LEN],
            "attention_mask": encoding["attention_mask"][:MAX_LEN],
            "labels": labels[:MAX_LEN],
        }


# ---------------------
# SPLIT
# ---------------------
split_idx = int(0.9 * len(dataset))
train_dataset = POSDataset(dataset[:split_idx])
val_dataset = POSDataset(dataset[split_idx:])


# ---------------------
# COLLATOR
# ---------------------
def collator(features):
    labels = [torch.tensor(f.pop("labels")) for f in features]
    batch = tokenizer.pad(features, return_tensors="pt")
    batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)
    return batch


# ---------------------
# METRICS
# ---------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    mask = labels != -100
    acc = accuracy_score(labels[mask], preds[mask])
    return {"accuracy": acc}


# ---------------------
# CUSTOM TRAINER (CONF MATRIX)
# ---------------------
class CMTrainer(Trainer):
    def evaluate(self, eval_dataset=None, **kwargs):
        metrics = super().evaluate(eval_dataset, **kwargs)

        preds_out = self.predict(eval_dataset)
        preds = np.argmax(preds_out.predictions, axis=-1)
        labels = preds_out.label_ids

        mask = labels != -100
        y_true = labels[mask]
        y_pred = preds[mask]

        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=UPOS
            ),
            "epoch": self.state.epoch
        })

        return metrics


# ---------------------
# HYPERPARAMETER GRID
# ---------------------
r_list = [4, 8, 16]
alpha_list = [8, 16]
dropout_list = [0.05, 0.1]
lr_list = [5e-5, 1e-4]

results = []


# ---------------------
# SWEEP LOOP
# ---------------------
for r in r_list:
    for alpha in alpha_list:
        for dropout in dropout_list:
            for lr in lr_list:

                run_name = f"r{r}_a{alpha}_d{dropout}_lr{lr}"

                wandb.init(
                    project="LoRA_POS_sweep",
                    name=run_name,
                    config={
                        "r": r,
                        "alpha": alpha,
                        "dropout": dropout,
                        "lr": lr,
                        "epochs": 5,
                        "batch_size": 8,
                    },
                    reinit=True
                )

                print(f"\n🚀 Training {run_name}\n")

                base_model = XLMRobertaForTokenClassification.from_pretrained(
                    "xlm-roberta-base",
                    num_labels=num_labels
                )

                lora_config = LoraConfig(
                    task_type=TaskType.TOKEN_CLS,
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    target_modules=["query", "key", "value"],
                    bias="none"
                )

                model = get_peft_model(base_model, lora_config)
                model.to(device)

                args = TrainingArguments(
                    output_dir=os.path.join(OUTPUT_DIR, run_name),
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    learning_rate=lr,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    num_train_epochs=3 if SMOKE_TEST else 5,
                    logging_strategy="steps",
                    logging_steps=50,
                    save_total_limit=1,
                    load_best_model_at_end=True,
                    report_to="wandb",
                    run_name=run_name
                )

                trainer = CMTrainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    compute_metrics=compute_metrics
                )

                trainer.train()
                eval_results = trainer.evaluate()

                results.append({
                    "r": r,
                    "alpha": alpha,
                    "dropout": dropout,
                    "learning_rate": lr,
                    "eval_accuracy": eval_results["eval_accuracy"],
                    "eval_loss": eval_results["eval_loss"]
                })

                wandb.finish()


# ---------------------
# SAVE RESULTS
# ---------------------
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "lora_sweep_results.csv"), index=False)

print("\n Hyperparameter sweep finished.")
