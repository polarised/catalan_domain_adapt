from tokenizer import token_label_alignment
from data_utils import load_pickle
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
import torch
import os
from sklearn.metrics import accuracy_score

# Load any clean dataset for tokenization
load_dir = r'C:\Users\mikel\OneDrive\Desktop\Uni\4th Year\1st Semester\Adv. Automatic Learning\Project\Code\Preprocessing\Clean_Datasets'
load_path = os.path.join(load_dir, "cleaned_ud_dataset(cat)_completo.pkl")

loaded_dataset = load_pickle(load_path)

# Visualize a sentence
tokens, tags = loaded_dataset[2]

for tok, tag in zip(tokens, tags):
    print(f"{tok:15s} ->  {tag}")

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

# Build UPOS vocabulary safely
all_tags = set()
for _, tags in loaded_dataset:
    for tag in tags:
        if tag is None:   # skip or replace None
            tag = "X"     # X is standard for unknown/other
        all_tags.add(tag)

UPOS = sorted(all_tags)
upos2id = {tag: i for i, tag in enumerate(UPOS)}
id2upos = {i: tag for tag, i in upos2id.items()}

# Load model AFTER we know num_labels
classifier = XLMRobertaForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(UPOS)
)
classifier.eval()

# --- Tokenize and align ---
encoding, aligned_labels_str = token_label_alignment(tokens, tags, tokenizer)

aligned_labels = [
    -100 if lbl == -100 or lbl is None else upos2id[lbl]
    for lbl in aligned_labels_str
]

subtokens = encoding.tokens()
word_ids = encoding.word_ids()

print(" ".join([f"{t}/{l}" if l != -100 else t for t, l in zip(subtokens, aligned_labels)]))
print(f"Number of subtokens: {len(encoding['input_ids'])}")

# --- Baseline prediction ---
input_ids = torch.tensor([encoding["input_ids"]])
attention_mask = torch.tensor([encoding["attention_mask"]])

with torch.no_grad():
    logits = classifier(input_ids, attention_mask=attention_mask).logits[0]

# Compare model prediction vs gold labels for visible tokens
gold = []
pred = []

for i, w in enumerate(word_ids):
    if w is None: 
        continue

    gold.append(aligned_labels[i])
    pred.append(int(logits[i].argmax()))


acc = accuracy_score(gold, pred)

print(f"\nBaseline zero-shot POS accuracy for this sentence: {acc:.6f}")