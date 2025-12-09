import os
import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
from collections import defaultdict

from tokenizer import token_label_alignment
from data_utils import load_pickle

# ---------------------
# CONFIG
# ---------------------
DATA_DIR = r'C:\Users\mikel\OneDrive\Desktop\Uni\4th Year\1st Semester\Adv. Automatic Learning\Project\Code\Preprocessing\Clean_Datasets'
DATA_FILE = "cleaned_ud_dataset(cat)_completo.pkl"

MODEL_DIR = "./pos_tagger_model"   # folder created by training script
MAX_ERRORS_TO_SHOW = 50

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
# TOKENIZER + MODEL
# ---------------------
tokenizer = XLMRobertaTokenizerFast.from_pretrained(MODEL_DIR)
model = XLMRobertaForTokenClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

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

# ---------------------
# ERROR TRACKERS
# ---------------------
confusion = defaultdict(int)
errors = []

# ---------------------
# MAIN LOOP
# ---------------------
for sent_idx, (tokens, tags) in enumerate(dataset):

    encoding, aligned_labels_str = token_label_alignment(tokens, tags, tokenizer)

    aligned_labels = [
        -100 if lbl == -100 or lbl is None else upos2id[lbl]
        for lbl in aligned_labels_str
    ]

    input_ids = torch.tensor([encoding["input_ids"]]).to(device)
    attention_mask = torch.tensor([encoding["attention_mask"]]).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits[0]

    word_ids = encoding.word_ids()

    prev_word = None
    for i, w_id in enumerate(word_ids):
        if w_id is None or w_id == prev_word:
            continue
        prev_word = w_id

        gold = aligned_labels[i]
        if gold == -100:
            continue

        pred = int(torch.argmax(logits[i]))

        if gold != pred:
            word = tokens[w_id]
            gold_tag = id2upos[gold]
            pred_tag = id2upos[pred]

            confusion[(gold_tag, pred_tag)] += 1
            errors.append((word, gold_tag, pred_tag, tokens))

# ---------------------
# PRINT COMMON CONFUSIONS
# ---------------------
print("\n=== Most common confusions ===\n")

sorted_conf = sorted(confusion.items(), key=lambda x: x[1], reverse=True)

for (gold, pred), count in sorted_conf[:30]:
    print(f"{gold:10s} → {pred:10s} : {count}")

# ---------------------
# PRINT EXAMPLES
# ---------------------
print("\n=== Example errors ===\n")

for i, (word, gold, pred, sentence) in enumerate(errors[:MAX_ERRORS_TO_SHOW]):
    print(f"Sentence: {' '.join(sentence)}")
    print(f"Word: {word}")
    print(f"Gold: {gold} | Pred: {pred}")
    print("-" * 50)

print(f"\nTotal errors found: {len(errors)}")
