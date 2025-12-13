#!/usr/bin/env python3
"""
mlm_train.py

Assumes the following classes are provided in another module:
  - WikiDownloader
  - WikiExtractor
  - WikiCleaner
  - TokenizerWrapper

Adjust CLI args as needed.
"""
import os
import math
import logging
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
    AutoTokenizer,
)
import torch

# Replace `my_preproc` with the module where your classes live
from data_processing import WikiDownloader, WikiExtractor, WikiCleaner, TokenizerWrapper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def group_texts(examples, block_size):
    """
    Concatenate texts and split into chunks of size block_size.
    Based on Hugging Face examples.group_texts
    """
    # Concatenate all input_ids (they are lists)
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    # Drop small remainder to make it divisible by block_size
    total_length = (total_length // block_size) * block_size
    result = {}
    for k, v in concatenated.items():
        # split into chunks of block_size
        chunks = [v[i : i + block_size] for i in range(0, total_length, block_size)]
        result[k] = chunks
    # Attention mask exists; if not, create full-mask chunks
    if "attention_mask" not in result:
        result["attention_mask"] = [[1] * block_size for _ in range(len(result["input_ids"]))]
    return result


def main():
    parser = argparse.ArgumentParser(description="Train XLM-R MLM on wiki dump")
    parser.add_argument("--dump-url", default="https://dumps.wikimedia.org/cawiki/latest/cawiki-latest-pages-articles.xml.bz2")
    parser.add_argument("--dump-file", default="cawiki-latest-pages-articles.xml.bz2")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--tokenizer-model", default="xlm-roberta-base")
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=128, help="tokenizer max_length (used for initial tokenization)")
    parser.add_argument("--block-size", type=int, default=128, help="size of chunks fed to MLM")
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--output-dir", default="./xlm_roberta_mlm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-group-texts", action="store_true", help="Concatenate and chunk texts (recommended)")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading dump if already present")
    parser.add_argument("--gradient-accumulation-steps",type=int,default=8,help="Number of gradient accumulation steps (default: 8)")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Download & extract (using your classes)
    downloader = WikiDownloader(args.dump_url, args.dump_file)
    if not args.no_download:
        downloader.download()
    cleaner = WikiCleaner()
    extractor = WikiExtractor(args.dump_file, args.sample_size, cleaner)
    documents = extractor.extract()
    if len(documents) == 0:
        raise SystemExit("No documents extracted. Check your dump path or sample size.")

    # 2) Build a Dataset
    ds = Dataset.from_list(documents)
    ds = ds.shuffle(seed=args.seed)

    # 3) Tokenizer - use the provided TokenizerWrapper or fallback to AutoTokenizer
    #    The TokenizerWrapper is expected to expose .tokenizer (HF tokenizer)
    try:
        tokenizer_wrapper = TokenizerWrapper(model_name=args.tokenizer_model, max_length=args.max_length)
        tokenizer = tokenizer_wrapper.tokenizer
    except Exception as e:
        logger.warning("TokenizerWrapper import/instantiation failed; falling back to AutoTokenizer. Error: %s", e)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)

    # Ensure tokenizer has mask token
    if tokenizer.mask_token is None:
        raise RuntimeError("Tokenizer does not have a mask token required for MLM.")

    # 4) Tokenize lines (return special tokens mask to let collator avoid masking special tokens)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=True,
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=["text"])

    # 5) Optionally group texts into contiguous blocks (better context utilization)
    if args.use_group_texts:
        # group_texts expects input_ids & optionally attention_mask
        tokenized = tokenized.map(lambda ex: group_texts(ex, args.block_size), batched=True, remove_columns=tokenized.column_names)

    # 6) Split train/eval
    split = tokenized.train_test_split(test_size=0.05, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    # 7) Set format for PyTorch
    columns = ["input_ids", "attention_mask"]
    # special_tokens_mask is not needed as a tensor (collator expects it in batch examples)
    if "special_tokens_mask" in train_ds.column_names:
        # keep it for the collator (map will include it in samples)
        columns = columns + ["special_tokens_mask"]
    train_ds.set_format(type="torch", columns=[c for c in columns if c in train_ds.column_names])
    eval_ds.set_format(type="torch", columns=[c for c in columns if c in eval_ds.column_names])

    # 8) Model
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    # 9) Data collator (creates masked labels on the fly). It respects special_tokens_mask if returned.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    # 10) TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
    )


    # 11) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 12) Train
    logger.info("Starting training")
    trainer.train()

    # 13) Save
    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 14) Final eval & perplexity
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        try:
            ppl = math.exp(metrics["eval_loss"])
        except OverflowError:
            ppl = float("inf")
        logger.info("Final eval_loss: %f  perplexity: %s", metrics["eval_loss"], str(ppl))

    logger.info("Done.")


if __name__ == "__main__":
    main()
