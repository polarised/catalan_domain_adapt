import pyconll
import pickle
from datasets import Dataset
from tokenizer import token_label_alignment


# This will load theconllu file and clean it into just token and POS-tag
def data_loader(file_path):
    dataset = pyconll.load_from_file(file_path)
    sentences = []

    for sentence in dataset:  # just first 10 for testing
        tokens = []
        tags = []
        for token in sentence:
            tokens.append(token.form)
            tags.append(token.upos)

        sentences.append((tokens, tags))

    return sentences

def save_pickle(dataset, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to: {file_path}")

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def to_hf_dataset(data, tokenizer, upos2id):
    """
    Convert tokenized + aligned data into a HuggingFace Dataset for token classification.
    Safely handles None and -100 tags.
    
    Args:
        data (list of tuples): Each tuple is (tokens, tags)
        tokenizer: HF tokenizer
        upos2id (dict): Mapping UPOS tag -> integer ID
    
    Returns:
        datasets.Dataset
    """
    encodings = []
    labels = []

    for tokens, tags in data:
        # Align subtokens
        encoding, aligned = token_label_alignment(tokens, tags, tokenizer)

        # Map tags to numeric IDs
        numeric_labels = [
            -100 if (l is None or l == -100) else upos2id[l]
            for l in aligned
        ]

        encodings.append(encoding)
        labels.append(numeric_labels)

    # Build HF Dataset
    hf_dataset = Dataset.from_dict({
        "input_ids": [e["input_ids"] for e in encodings],
        "attention_mask": [e["attention_mask"] for e in encodings],
        "labels": labels
    })

    return hf_dataset