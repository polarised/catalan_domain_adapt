import os
import torch
from torch.utils.data import Dataset
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
    Trainer,
    TrainingArguments
)

from tokenizer import token_label_alignment
from data_utils import load_pickle

# ---------------------
# CONFIG
# ---------------------
DATA_DIR = r'C:\Users\mikel\OneDrive\Desktop\Uni\4th Year\1st Semester\Adv. Automatic Learning\Project\Code\Preprocessing\Clean_Datasets'
DATA_FILE = "cleaned_ud_dataset(cat)_completo.pkl"
OUTPUT_DIR = "./pos_tagger_model"

EPOCHS = 3
BATCH_SIZE = 8
LR = 5e-5
MAX_LEN = 128

# ---------------------
# DEVICE
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------
# LOAD DATA
# ---------------------
data_path = os.path.join(DATA_DIR, DATA_FILE)
dataset = load_pickle(data_path)

print(f"Loaded {len(dataset)} sentences")

# ---------------------
# TOKENIZER
# ---------------------
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

# ---------------------
# BUILD TAG VOCAB
# ---------------------
all_tags = set()
for _, tags in dataset:
    for tag in tags:
        if tag is None:
            tag = "X"
        all_tags.add(tag)

UPOS = sorted(all_tags)
upos2id = {tag: i for i, tag in enumerate(UPOS)}
id2upos = {i: tag for tag, i in upos2id.items()}

num_labels = len(UPOS)
print("Number of labels:", num_labels)

# ---------------------
# DATASET CLASS
# ---------------------
class POSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tags = self.data[idx]

        encoding, aligned_labels_str = token_label_alignment(
            tokens, tags, tokenizer
        )

        aligned_labels = [
            -100 if lbl == -100 or lbl is None else upos2id[lbl]
            for lbl in aligned_labels_str
        ]

        item = {
            "input_ids": torch.tensor(encoding["input_ids"][:MAX_LEN]),
            "attention_mask": torch.tensor(encoding["attention_mask"][:MAX_LEN]),
            "labels": torch.tensor(aligned_labels[:MAX_LEN])
        }

        return item

# ---------------------
# SPLIT DATA
# ---------------------
split_idx = int(0.9 * len(dataset))
train_data = dataset[:split_idx]
val_data = dataset[split_idx:]

train_dataset = POSDataset(train_data)
val_dataset = POSDataset(val_data)

# ---------------------
# MODEL
# ---------------------
model = XLMRobertaForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=num_labels
)
model.to(device)

# ---------------------
# TRAINING ARGS
# ---------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True
)

# ---------------------
# TRAINER
# ---------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# ---------------------
# START TRAINING
# ---------------------
print("\nStarting training...\n")
trainer.train()

# ---------------------
# SAVE MODEL
# ---------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete. Model saved to:", OUTPUT_DIR)
