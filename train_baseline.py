from data_utils import load_pickle, to_hf_dataset
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
import os
import torch
import numpy as np
import json
import time

# -------------------------
# 0) GPU setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# -------------------------
# 1) Load cleaned datasets
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(script_dir, "clean_Datasets")
output_dir="./Training/model_output"
logging_dir="./Training/logs"

print("\nLoading datasets...")
train_data = load_pickle(os.path.join(BASE, "cleaned_ud_dataset(cat)_completo_train.pkl"))
dev_data   = load_pickle(os.path.join(BASE, "cleaned_ud_dataset(cat)_completo_dev.pkl"))
test_data  = load_pickle(os.path.join(BASE, "cleaned_ud_dataset(cat)_completo.pkl"))

print(f"Loaded {len(train_data)} training sentences")
print(f"Loaded {len(dev_data)} dev sentences")
print(f"Loaded {len(test_data)} test sentences")

# -------------------------
# 2) Build label mappings from ALL data (train + dev + test)
# -------------------------
print("\nBuilding POS tag vocabulary from all datasets...")
all_tags = set()

# Collect tags from all datasets to ensure complete vocabulary
for dataset in [train_data, dev_data, test_data]:
    for tokens, tags in dataset:
        for tag in tags:
            if tag is not None and tag != -100:
                all_tags.add(tag)

UPOS = sorted(all_tags)
upos2id = {t: i for i, t in enumerate(UPOS)}
id2upos = {i: t for t, i in upos2id.items()}

print(f"Number of POS tags: {len(UPOS)}")
print(f"POS tags: {UPOS}")

# Check for tags in test/dev that aren't in train
train_tags = {tag for _, tags in train_data for tag in tags if tag not in [None, -100]}
test_tags = {tag for _, tags in test_data for tag in tags if tag not in [None, -100]}
dev_tags = {tag for _, tags in dev_data for tag in tags if tag not in [None, -100]}

unseen_in_train = (test_tags | dev_tags) - train_tags
if unseen_in_train:
    print(f"\n⚠ Warning: Tags in test/dev but not in train: {unseen_in_train}")
    print("   Model will have limited exposure to these tags.")

# -------------------------
# 3) Tokenizer + model
# -------------------------
print("\nLoading tokenizer and model...")
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
model = XLMRobertaForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(UPOS)
).to(device)

# Add label mappings to model config
model.config.id2label = id2upos
model.config.label2id = upos2id

print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# -------------------------
# 4) Convert datasets to HF format
# -------------------------
print("\nConverting datasets to HuggingFace format...")
max_seq_length = 512

train_ds = to_hf_dataset(train_data, tokenizer, upos2id)
dev_ds   = to_hf_dataset(dev_data, tokenizer, upos2id)
test_ds  = to_hf_dataset(test_data, tokenizer, upos2id)

print(f"Train dataset: {len(train_ds)} samples")
print(f"Dev dataset: {len(dev_ds)} samples")
print(f"Test dataset: {len(test_ds)} samples")

# -------------------------
# 5) Metrics computation
# -------------------------
def compute_metrics(pred):
    """Compute accuracy and macro F1 for POS tagging."""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens with label -100)
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        true_pred = []
        true_lab = []
        for p, l in zip(prediction, label):
            if l != -100:  # Only consider actual tokens, not special tokens
                true_pred.append(id2upos[p])
                true_lab.append(id2upos[l])
        
        if true_pred:  # Only add if there are actual tokens
            true_predictions.append(true_pred)
            true_labels.append(true_lab)
    
    # Calculate overall accuracy
    correct = sum(p == l for preds, labs in zip(true_predictions, true_labels) 
                  for p, l in zip(preds, labs))
    total = sum(len(labs) for labs in true_labels)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate per-tag metrics for macro F1
    tag_stats = {tag: {"tp": 0, "fp": 0, "fn": 0} for tag in UPOS}
    
    for preds, labs in zip(true_predictions, true_labels):
        for p, l in zip(preds, labs):
            if p == l:
                tag_stats[l]["tp"] += 1
            else:
                tag_stats[p]["fp"] += 1
                tag_stats[l]["fn"] += 1
    
    # Calculate macro F1
    f1_scores = []
    precisions = []
    recalls = []
    
    for tag, stats in tag_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    
    macro_f1 = np.mean(f1_scores) if f1_scores else 0
    macro_precision = np.mean(precisions) if precisions else 0
    macro_recall = np.mean(recalls) if recalls else 0
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }

# -------------------------
# 6) TrainingArguments
# -------------------------
batch_size = 16 if torch.cuda.is_available() else 8

training_args = TrainingArguments(
    output_dir="./pruebas/Training/model_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,  # Keep only 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    logging_dir="./pruebas/Training/logs",
    logging_strategy="steps",
    logging_steps=100,
    logging_first_step=True,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,  # Larger batch for eval
    num_train_epochs=5,  # More epochs for better convergence
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_num_workers=0,  # Avoid Windows multiprocessing issues
    report_to="tensorboard",  # Enable tensorboard logging
    seed=42,
    max_grad_norm=1.0,
    disable_tqdm=False,
)

# -------------------------
# 7) Data Collator
# -------------------------
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
)

# -------------------------
# 8) Trainer with early stopping
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# -------------------------
# 9) Train
# -------------------------
print("\n" + "="*70)
print("STARTING TRAINING - XLM-RoBERTa for Catalan POS Tagging")
print("="*70)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(dev_ds)}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Learning rate: {training_args.learning_rate}")
print("="*70 + "\n")

start_time = time.time()

try:
    trainer.train()
    training_time = time.time() - start_time
    print(f"\n✓ Training completed successfully in {training_time/60:.2f} minutes!")
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    raise

# -------------------------
# 10) Evaluate on dev and test sets
# -------------------------
print("\n" + "="*70)
print("EVALUATION ON DEVELOPMENT SET")
print("="*70)
dev_metrics = trainer.evaluate(eval_dataset=dev_ds)
print("\nDevelopment Set Results:")
print(f"  Accuracy:        {dev_metrics['eval_accuracy']:.4f} ({dev_metrics['eval_accuracy']*100:.2f}%)")
print(f"  Macro F1:        {dev_metrics['eval_macro_f1']:.4f}")
print(f"  Macro Precision: {dev_metrics['eval_macro_precision']:.4f}")
print(f"  Macro Recall:    {dev_metrics['eval_macro_recall']:.4f}")
print(f"  Loss:            {dev_metrics['eval_loss']:.4f}")

print("\n" + "="*70)
print("EVALUATION ON TEST SET")
print("="*70)
test_start = time.time()
test_metrics = trainer.evaluate(eval_dataset=test_ds)
test_time = time.time() - test_start

print("\nTest Set Results:")
print(f"  Accuracy:        {test_metrics['eval_accuracy']:.4f} ({test_metrics['eval_accuracy']*100:.2f}%)")
print(f"  Macro F1:        {test_metrics['eval_macro_f1']:.4f}")
print(f"  Macro Precision: {test_metrics['eval_macro_precision']:.4f}")
print(f"  Macro Recall:    {test_metrics['eval_macro_recall']:.4f}")
print(f"  Loss:            {test_metrics['eval_loss']:.4f}")
print(f"\nTest evaluation time: {test_time:.1f} seconds")

# -------------------------
# 11) Calculate per-tag performance on test set
# -------------------------
print("\n" + "="*70)
print("PER-TAG PERFORMANCE ON TEST SET")
print("="*70)

# Get predictions for test set
predictions = trainer.predict(test_ds)
pred_labels = np.argmax(predictions.predictions, axis=2)

# Calculate per-tag metrics
tag_stats = {tag: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for tag in UPOS}

for pred, label in zip(pred_labels, predictions.label_ids):
    for p, l in zip(pred, label):
        if l != -100:
            true_tag = id2upos[l]
            pred_tag = id2upos[p]
            tag_stats[true_tag]["support"] += 1
            
            if pred_tag == true_tag:
                tag_stats[true_tag]["tp"] += 1
            else:
                tag_stats[pred_tag]["fp"] += 1
                tag_stats[true_tag]["fn"] += 1

# Print per-tag results
print(f"\n{'Tag':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
print("-" * 60)

for tag in UPOS:
    stats = tag_stats[tag]
    tp, fp, fn, support = stats["tp"], stats["fp"], stats["fn"], stats["support"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{tag:<8} {precision:>6.2%}       {recall:>6.2%}       {f1:>6.2%}       {support:<10}")

# -------------------------
# 12) Save everything
# -------------------------
print("\n" + "="*70)
print("SAVING MODEL AND RESULTS")
print("="*70)

trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

# Save label mappings
label_mapping_path = os.path.join(training_args.output_dir, "label_mapping.json")
with open(label_mapping_path, "w", encoding="utf-8") as f:
    json.dump({
        "id2label": id2upos,
        "label2id": upos2id,
        "num_labels": len(UPOS),
        "pos_tags": UPOS
    }, f, ensure_ascii=False, indent=2)

# Save comprehensive metrics
metrics_path = os.path.join(training_args.output_dir, "evaluation_results.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({
        "dev_metrics": {
            "accuracy": float(dev_metrics['eval_accuracy']),
            "macro_f1": float(dev_metrics['eval_macro_f1']),
            "macro_precision": float(dev_metrics['eval_macro_precision']),
            "macro_recall": float(dev_metrics['eval_macro_recall']),
            "loss": float(dev_metrics['eval_loss'])
        },
        "test_metrics": {
            "accuracy": float(test_metrics['eval_accuracy']),
            "macro_f1": float(test_metrics['eval_macro_f1']),
            "macro_precision": float(test_metrics['eval_macro_precision']),
            "macro_recall": float(test_metrics['eval_macro_recall']),
            "loss": float(test_metrics['eval_loss'])
        },
        "per_tag_metrics": {
            tag: {
                "precision": float(stats["tp"] / (stats["tp"] + stats["fp"])) if (stats["tp"] + stats["fp"]) > 0 else 0,
                "recall": float(stats["tp"] / (stats["tp"] + stats["fn"])) if (stats["tp"] + stats["fn"]) > 0 else 0,
                "f1": float(2 * stats["tp"] / (2 * stats["tp"] + stats["fp"] + stats["fn"])) if (2 * stats["tp"] + stats["fp"] + stats["fn"]) > 0 else 0,
                "support": int(stats["support"])
            }
            for tag, stats in tag_stats.items()
        },
        "training_info": {
            "model": "xlm-roberta-base",
            "train_samples": len(train_ds),
            "dev_samples": len(dev_ds),
            "test_samples": len(test_ds),
            "batch_size": batch_size,
            "learning_rate": training_args.learning_rate,
            "num_epochs": training_args.num_train_epochs,
            "max_sequence_length": max_seq_length,
            "training_time_minutes": training_time / 60,
            "device": str(device),
        }
    }, f, ensure_ascii=False, indent=2)

print(f"\n✓ Model saved to: {training_args.output_dir}")
print(f"✓ Tokenizer saved to: {training_args.output_dir}")
print(f"✓ Label mappings saved to: {label_mapping_path}")
print(f"✓ Evaluation results saved to: {metrics_path}")
print(f"✓ TensorBoard logs saved to: {training_args.logging_dir}")

print("\n" + "="*70)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nTraining time: {training_time/60:.2f} minutes")
print(f"Final Test Accuracy: {test_metrics['eval_accuracy']*100:.2f}%")
print(f"Final Test Macro F1: {test_metrics['eval_macro_f1']:.4f}")
print(f"\nTo view training logs, run: tensorboard --logdir={training_args.logging_dir}")
print("="*70)