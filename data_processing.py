import os
import bz2
import random
import requests
import numpy as np
from tqdm import tqdm
import mwxml
import mwparserfromhell
from datasets import Dataset
from transformers import AutoTokenizer


class WikiDownloader:
    def __init__(self, url: str, out_file: str):
        self.url = url
        self.out_file = out_file

    def download(self):
        if os.path.exists(self.out_file):
            print("Dump already exists, skipping download.")
            return

        print("Downloading Wikipedia dump...")
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            with open(self.out_file, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192)):
                    f.write(chunk)


class WikiCleaner:
    def clean(self, text: str) -> str:
        """Remove wiki markup, templates, categories, etc."""
        wikicode = mwparserfromhell.parse(text)
        text = wikicode.strip_code()
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text


class WikiExtractor:
    def __init__(self, dump_file: str, sample_size: int, cleaner: WikiCleaner):
        self.dump_file = dump_file
        self.sample_size = sample_size
        self.cleaner = cleaner

    def extract(self):
        print("Extracting pages...")
        documents = []

        with bz2.open(self.dump_file, "rb") as f:
            dump = mwxml.Dump.from_file(f)

            for i, page in enumerate(dump.pages):
                if i >= self.sample_size:
                    break
                if page.redirect:
                    continue

                for revision in page:
                    if revision.text and len(revision.text) > 100:
                        cleaned = self.cleaner.clean(revision.text)
                        if len(cleaned) > 50:
                            documents.append({"text": cleaned})
                    break

        print(f"Total documents extracted: {len(documents)}")
        return documents


class TokenizerWrapper:
    def __init__(self, model_name="bert-base-multilingual-cased", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize_fn(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def tokenize_dataset(self, dataset: Dataset):
        print("Tokenizing dataset...")
        return dataset.map(
            self.tokenize_fn,
            batched=True,
            remove_columns=["text"]
        )

    def compute_stats(self, tokenized_ds):
        lengths = [
            sum(x != self.tokenizer.pad_token_id for x in ex["input_ids"])
            for ex in tokenized_ds
        ]

        return {
            "count": len(lengths),
            "avg_tokens": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "max": int(np.max(lengths)),
            "min": int(np.min(lengths)),
        }


class CatalanWikiPipeline:
    def __init__(
        self,
        dump_url="https://dumps.wikimedia.org/cawiki/latest/cawiki-latest-pages-articles.xml.bz2",
        dump_file="cawiki-latest-pages-articles.xml.bz2",
        sample_size=5000,
        tokenizer_model="bert-base-multilingual-cased",
        max_length=128,
    ):
        self.downloader = WikiDownloader(dump_url, dump_file)
        self.cleaner = WikiCleaner()
        self.extractor = WikiExtractor(dump_file, sample_size, self.cleaner)
        self.tokenizer = TokenizerWrapper(tokenizer_model, max_length)

    def run(self):
        self.downloader.download()
        documents = self.extractor.extract()

        ds = Dataset.from_list(documents)
        tokenized = self.tokenizer.tokenize_dataset(ds)

        stats = self.tokenizer.compute_stats(tokenized)
        print("=== Token Statistics ===")
        for k, v in stats.items():
            print(f"{k}: {v}")

        return tokenized, stats

if __name__ == "__main__":
    pipeline = CatalanWikiPipeline(
        sample_size=5000,
        max_length=128,
    )
    tokenized, stats = pipeline.run()
