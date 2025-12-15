# =====================
# LoRA POS SWEEP (ADVANCED)
# =====================

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_scheduler
)

from peft import LoraConfig, TaskType, get_peft_model

from tokenizer import token_label_alignment
from data_utils import load_pickle
import wandb

# ---------------------
# PATHS (CLUSTER SAFE)
# ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Clean_Datasets", "cleaned_ud_dataset(cat)_prueba1000.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "pos_tagger_lora_sweep_advanced")
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
all_tags = set(tag if tag is not None else "X" for _, tags in dataset for tag in tags)
UPOS = sorted(all_tags)
upos2id = {t: i for i, t in enumerate(UPOS)}
id2upos = {i: t for t, i in upos2id.items()}
num_labels = len(UPOS)

# ---------------------
# DATASET
# ---------------------
class POSDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def augment_tokens(self, tokens):
        """Optional simple augmentation: random swap of non-adjacent words"""
        tokens = tokens.copy()
        if len(tokens) > 3:
            i, j = np.random.choice(len(tokens), 2, replace=False)
            tokens[i], tokens[j] = tokens[j], tokens[i]
        return tokens

    def __getitem__(self, idx):
        tokens, tags = self.data[idx]

        if self.augment:
            tokens = self.augment_tokens(tokens)

        encoding, aligned = token_label_alignment(tokens, tags, tokenizer)
        labels = [-100 if l in (-100, None) else upos2id.get(l, -100) for l in aligned]

        return {
            "input_ids": encoding["input_ids"][:MAX_LEN],
            "attention_mask": encoding["attention_mask"][:MAX_LEN],
            "labels": labels[:MAX_LEN],
        }

# ---------------------
# SPLIT DATA
# ---------------------
split_idx = int(0.9 * len(dataset))
train_dataset = POSDataset(dataset[:split_idx], augment=True)
val_dataset = POSDataset(dataset[split_idx:])

# ---------------------
# COLLATOR
# ---------------------
def collator(features):
    labels = [torch.tensor(f.pop("labels"), dtype=torch.long) for f in features]
    batch = tokenizer.pad(features, return_tensors="pt")
    seq_len = batch["input_ids"].shape[1]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    batch["labels"] = labels_padded[:, :seq_len]
    return batch

# ---------------------
# METRICS
# ---------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    mask = labels != -100
    y_true = labels[mask]
    y_pred = preds[mask]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

# ---------------------
# CUSTOM TRAINER
# ---------------------
class WandBCMTrainer(Trainer):
    def evaluate(self, eval_dataset=None, **kwargs):
        metrics = super().evaluate(eval_dataset, **kwargs)
        preds_out = self.predict(eval_dataset)
        preds = np.argmax(preds_out.predictions, axis=-1)
        labels = preds_out.label_ids
        mask = labels != -100
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=labels[mask],
                preds=preds[mask],
                class_names=UPOS
            ),
            "eval_accuracy": metrics.get("eval_accuracy", 0),
            "eval_f1_macro": metrics.get("eval_f1_macro", 0),
            "eval_loss": metrics.get("eval_loss", 0),
            "epoch": self.state.epoch
        })
        return metrics

# ---------------------
# HYPERPARAM GRID
# ---------------------
r_list = [8, 16]
alpha_list = [16, 32]
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
                    project="LoRA_POS_sweep_advanced",
                    name=run_name,
                    config={
                        "r": r,
                        "alpha": alpha,
                        "dropout": dropout,
                        "lr": lr,
                        "epochs": 5,
                        "batch_size": 8,
                        "max_len": MAX_LEN
                    },
                    reinit=True
                )

                print(f"\n🚀 Training {run_name}\n")

                # ---------------------
                # MODEL + LoRA
                # ---------------------
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

                # ---------------------
                # TRAINING ARGS (ADVANCED)
                # ---------------------
                args = TrainingArguments(
                    output_dir=os.path.join(OUTPUT_DIR, run_name),
                    eval_strategy="steps",
                    save_strategy="epoch",
                    logging_strategy="steps",
                    logging_steps=50,
                    learning_rate=lr,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    num_train_epochs=5,
                    save_total_limit=2,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1_macro",
                    greater_is_better=True,
                    report_to="wandb",
                    run_name=run_name,
                    gradient_accumulation_steps=2,   # simulates larger batch size
                    fp16=True,                        # mixed precision
                    lr_scheduler_type="cosine",       # smoother learning rate
                    warmup_steps=50
                )

                trainer = WandBCMTrainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                )

                # ---------------------
                # TRAIN
                # ---------------------
                trainer.train()
                eval_results = trainer.evaluate()

                results.append({
                    "r": r,
                    "alpha": alpha,
                    "dropout": dropout,
                    "learning_rate": lr,
                    "eval_accuracy": eval_results.get("eval_accuracy"),
                    "eval_f1_macro": eval_results.get("eval_f1_macro"),
                    "eval_loss": eval_results.get("eval_loss")
                })

                wandb.finish()

# ---------------------
# SAVE RESULTS
# ---------------------
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "lora_sweep_results_advanced.csv"), index=False)
print("\nAdvanced hyperparameter sweep finished.")
