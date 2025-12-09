import os
import torch
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
    AutoModelForTokenClassification
)
from sklearn.metrics import accuracy_score, classification_report

from tokenizer import token_label_alignment
from data_utils import load_pickle

LOAD_DIR = r'C:\Users\mikel\OneDrive\Desktop\Uni\4th Year\1st Semester\Adv. Automatic Learning\Project\Code\Preprocessing\Clean_Datasets'
DATA_FILE = "cleaned_ud_dataset(cat)_completo.pkl"

MODEL_NAME = "xlm-roberta-base"
# MODEL_NAME = "Davlan/xlm-roberta-base-ner-hrl"

USE_PRETRAINED_HEAD = False  #True if using a task-trained model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

load_path = os.path.join(LOAD_DIR, DATA_FILE)
dataset = load_pickle(load_path)
print(f"Loaded {len(dataset)} sentences")

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

all_tags = set()
for _, tags in dataset:
    for tag in tags:
        if tag is None:
            tag = "X"
        all_tags.add(tag)

UPOS = sorted(all_tags)
upos2id = {tag: i for i, tag in enumerate(UPOS)}
id2upos = {i: tag for tag, i in upos2id.items()}

print(f"Number of UPOS tags: {len(UPOS)}")
print("Tags:", UPOS)

if USE_PRETRAINED_HEAD:
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
else:
    model = XLMRobertaForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(UPOS)
    )

model.to(device)
model.eval()

all_gold = []
all_pred = []

print("\nRunning full-dataset baseline evaluation...")

for sent_idx, (tokens, tags) in enumerate(dataset):
    if len(tokens) == 0:
        continue

    # Tokenize and align
    encoding, aligned_labels_str = token_label_alignment(tokens, tags, tokenizer)

    aligned_labels = [
        -100 if lbl == -100 or lbl is None else upos2id[lbl]
        for lbl in aligned_labels_str
    ]

    input_ids = torch.tensor([encoding["input_ids"]]).to(device)
    attention_mask = torch.tensor([encoding["attention_mask"]]).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]

    word_ids = encoding.word_ids()

    # Word-level evaluation
    previous_word = None
    for i, w_id in enumerate(word_ids):
        if w_id is None or w_id == previous_word:
            continue
        previous_word = w_id

        gold_label = aligned_labels[i]
        if gold_label == -100:
            continue

        pred_label = int(torch.argmax(logits[i]))

        all_gold.append(gold_label)
        all_pred.append(pred_label)

    if (sent_idx + 1) % 500 == 0:
        print(f"Processed {sent_idx + 1}/{len(dataset)} sentences")

acc = accuracy_score(all_gold, all_pred)
print("\n===============================")
print(f"FINAL BASELINE WORD-LEVEL ACCURACY: {acc:.4f}")
print("===============================\n")

print("Classification report:\n")
print(
    classification_report(
        all_gold,
        all_pred,
        target_names=UPOS,
        digits=4
    )
)