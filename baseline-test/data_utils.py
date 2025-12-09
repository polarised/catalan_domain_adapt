import pyconll
import pickle

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


