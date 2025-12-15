# =====================
# LoRA POS TRAINING 
# =====================

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from tokenizer import token_label_alignment
from data_utils import load_pickle
import wandb

# ---------------------
# PATHS (CLUSTER SAFE)
# ---------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Clean_Datasets", "cleaned_ud_dataset(cat)_completo_train.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "lora_training", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 128
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-4

# ---------------------
# DEVICE
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------
# LOAD DATA
# ---------------------
dataset = load_pickle(DATA_PATH)
if dataset is None or len(dataset) == 0:
    raise ValueError(f"Dataset vacío o no cargado: {DATA_PATH}")
print(f"Dataset cargado correctamente. Total de ejemplos: {len(dataset)}")

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
num_labels = len(UPOS)
print(f"Etiqueta vocab size: {num_labels}")

# ---------------------
# DATASET
# ---------------------
class POSDataset(Dataset):
    def __init__(self, data):
        if data is None or len(data) == 0:
            raise ValueError("Dataset pasado a POSDataset vacío")
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tags = self.data[idx]
        encoding, aligned = token_label_alignment(tokens, tags, tokenizer)
        if encoding is None or aligned is None:
            raise ValueError(f"token_label_alignment devolvió None en idx {idx}")
        labels = [-100 if l in (-100, None) else upos2id.get(l, -100) for l in aligned]
        return {
            "input_ids": encoding["input_ids"][:MAX_LEN],
            "attention_mask": encoding["attention_mask"][:MAX_LEN],
            "labels": labels[:MAX_LEN],
        }

# ---------------------
# SPLIT DATA
# ---------------------
split = int(0.9 * len(dataset))
train_dataset = POSDataset(dataset[:split])
val_dataset = POSDataset(dataset[split:])
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

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
    acc = accuracy_score(labels[mask], preds[mask])
    return {"accuracy": acc}

# ---------------------
# MODEL + LoRA
# ---------------------
base_model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model.to(device)

# ---------------------
# WANDB INIT
# ---------------------
wandb.init(
    project="LoRA_POS_training",
    name="lora_experiment_v1",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_len": MAX_LEN,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
    }
)

# ---------------------
# TRAINING ARGS
# ---------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="wandb"
)

# ---------------------
# CUSTOM TRAINER (LOSS + ACC LOG)
# ---------------------
class WandBTrainer(Trainer):
    def log(self, logs):
        super().log(logs)
        # Registra loss y accuracy por step o epoch en wandb
        wandb.log({k: v for k, v in logs.items() if k in ["loss", "eval_loss", "eval_accuracy"]})

trainer = WandBTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# ---------------------
# TRAIN
# ---------------------
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished")
wandb.finish()

