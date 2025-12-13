from data_utils import load_pickle, to_hf_dataset
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import os
import torch
import numpy as np
import json
import time
from typing import Dict, List

# -------------------------
# 0) Configuration
# -------------------------

## XLM-R base has 12 layers, so it is easy to freeze a desired amount of them

EXPERIMENTS = {
    "full_finetuning": {
        "freeze_percentage": 0,
        "description": "Full fine-tuning (baseline)",
        "freeze_layers": None
    },
    "freeze_50": {
        "freeze_percentage": 50,
        "description": "Freeze bottom 50% of layers",
        "freeze_layers": [0, 1, 2, 3, 4, 5]  
    },
    "freeze_75": {
        "freeze_percentage": 75,
        "description": "Freeze bottom 75% of layers (train only top 3)",
        "freeze_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8]
    },
    "freeze_100": {
        "freeze_percentage": 100,
        "description": "Freeze all encoder layers (train only classifier)",
        "freeze_layers": "all"
    }
}

CURRENT_EXPERIMENT = "freeze_50"  # Can be changed to: "full_finetuning", "freeze_50", "freeze_75", "freeze_100"

# -------------------------
# 1) GPU setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# -------------------------
# 2) Load cleaned datasets
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(script_dir, "clean_Datasets")

print("\nLoading datasets...")
train_data = load_pickle(os.path.join(BASE, "cleaned_ud_dataset(cat)_completo_train.pkl"))
dev_data   = load_pickle(os.path.join(BASE, "cleaned_ud_dataset(cat)_completo_dev.pkl"))
test_data  = load_pickle(os.path.join(BASE, "cleaned_ud_dataset(cat)_completo.pkl"))

print(f"Loaded {len(train_data)} training sentences")
print(f"Loaded {len(dev_data)} dev sentences")
print(f"Loaded {len(test_data)} test sentences")

# -------------------------
# 3) Build label mappings
# -------------------------
print("\nBuilding POS tag vocabulary from all datasets...")
all_tags = set()

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

# -------------------------
# 4) Layer Freezing Function
# -------------------------
def freeze_layers(model, freeze_config):
    """
    Freeze specified layers in the XLM-RoBERTa model.
    
    Args:
        model: XLMRobertaForTokenClassification model
        freeze_config: Either a list of layer indices to freeze, or "all" to freeze everything
    """
    # First, unfreeze everything (in case we're reusing a model)
    for param in model.parameters():
        param.requires_grad = True
    
    if freeze_config is None:
        # Full fine-tuning - nothing to freeze
        return
    
    if freeze_config == "all":
        # Freeze all encoder layers (only train classifier)
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        return
    
    # Freeze specific layers
    for layer_idx in freeze_config:
        for name, param in model.named_parameters():
            if f"roberta.encoder.layer.{layer_idx}." in name:
                param.requires_grad = False
    
    # Always freeze embeddings for partial freezing experiments
    if freeze_config != "all" and len(freeze_config) > 0:
        for name, param in model.named_parameters():
            if "embeddings" in name:
                param.requires_grad = False

def count_parameters(model):
    """Count trainable and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_percentage": (trainable_params / total_params) * 100
    }

# -------------------------
# 5) Tokenizer + model
# -------------------------
print("\n" + "="*70)
print(f"EXPERIMENT: {EXPERIMENTS[CURRENT_EXPERIMENT]['description']}")
print(f"Freeze Percentage: {EXPERIMENTS[CURRENT_EXPERIMENT]['freeze_percentage']}%")
print("="*70)

print("\nLoading tokenizer and model...")
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
model = XLMRobertaForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(UPOS)
).to(device)

# Add label mappings to model config
model.config.id2label = id2upos
model.config.label2id = upos2id

# Apply layer freezing
print("\nApplying layer freezing strategy...")
freeze_layers(model, EXPERIMENTS[CURRENT_EXPERIMENT]["freeze_layers"])

# Count parameters
param_stats = count_parameters(model)
print(f"\nParameter Statistics:")
print(f"  Total parameters:      {param_stats['total']:,}")
print(f"  Trainable parameters:  {param_stats['trainable']:,} ({param_stats['trainable_percentage']:.2f}%)")
print(f"  Frozen parameters:     {param_stats['frozen']:,}")

# -------------------------
# 6) Convert datasets to HF format
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
# 7) Metrics computation
# -------------------------
def compute_metrics(pred):
    """Compute accuracy and macro F1 for POS tagging."""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        true_pred = []
        true_lab = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_pred.append(id2upos[p])
                true_lab.append(id2upos[l])
        
        if true_pred:
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
# 8) TrainingArguments
# -------------------------
batch_size = 16 if torch.cuda.is_available() else 8

# Create experiment-specific output directory
output_dir = f"./Training/layer_freezing_{CURRENT_EXPERIMENT}"
logging_dir = f"./Training/logs_{CURRENT_EXPERIMENT}"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,  # Keep only best checkpoint to save space
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    logging_dir=logging_dir,
    logging_strategy="steps",
    logging_steps=100,
    logging_first_step=True,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    num_train_epochs=3,  # Reduced for faster experiments
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_num_workers=0,
    report_to="tensorboard",
    seed=42,
    max_grad_norm=1.0,
    disable_tqdm=False,
)

# -------------------------
# 9) Data Collator and Trainer
# -------------------------
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# -------------------------
# 10) Train
# -------------------------
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(dev_ds)}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Learning rate: {training_args.learning_rate}")
print("="*70 + "\n")

# Measure GPU memory before training
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / 1e9

start_time = time.time()

try:
    trainer.train()
    training_time = time.time() - start_time
    
    # Measure peak GPU memory
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n✓ Training completed in {training_time/60:.2f} minutes!")
        print(f"✓ Peak GPU memory usage: {peak_memory:.2f} GB")
    else:
        print(f"\n✓ Training completed in {training_time/60:.2f} minutes!")
        peak_memory = 0
        
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    raise

# -------------------------
# 11) Evaluate
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
test_metrics = trainer.evaluate(eval_dataset=test_ds)
print("\nTest Set Results:")
print(f"  Accuracy:        {test_metrics['eval_accuracy']:.4f} ({test_metrics['eval_accuracy']*100:.2f}%)")
print(f"  Macro F1:        {test_metrics['eval_macro_f1']:.4f}")
print(f"  Macro Precision: {test_metrics['eval_macro_precision']:.4f}")
print(f"  Macro Recall:    {test_metrics['eval_macro_recall']:.4f}")
print(f"  Loss:            {test_metrics['eval_loss']:.4f}")

# -------------------------
# 12) Save everything
# -------------------------
print("\n" + "="*70)
print("SAVING MODEL AND RESULTS")
print("="*70)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Save comprehensive results
results_path = os.path.join(output_dir, "experiment_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({
        "experiment_name": CURRENT_EXPERIMENT,
        "experiment_config": EXPERIMENTS[CURRENT_EXPERIMENT],
        "parameter_stats": param_stats,
        "training_time_minutes": training_time / 60,
        "training_time_seconds": training_time,
        "peak_gpu_memory_gb": peak_memory if torch.cuda.is_available() else None,
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
        "training_config": {
            "model": "xlm-roberta-base",
            "train_samples": len(train_ds),
            "dev_samples": len(dev_ds),
            "test_samples": len(test_ds),
            "batch_size": batch_size,
            "learning_rate": training_args.learning_rate,
            "num_epochs": training_args.num_train_epochs,
            "device": str(device),
        }
    }, f, ensure_ascii=False, indent=2)

# Save label mappings
label_mapping_path = os.path.join(output_dir, "label_mapping.json")
with open(label_mapping_path, "w", encoding="utf-8") as f:
    json.dump({
        "id2label": id2upos,
        "label2id": upos2id,
        "num_labels": len(UPOS),
        "pos_tags": UPOS
    }, f, ensure_ascii=False, indent=2)

print(f"\n✓ Model saved to: {output_dir}")
print(f"✓ Results saved to: {results_path}")
print(f"✓ Label mappings saved to: {label_mapping_path}")

print("\n" + "="*70)
print("EXPERIMENT SUMMARY")
print("="*70)
print(f"Experiment: {EXPERIMENTS[CURRENT_EXPERIMENT]['description']}")
print(f"Trainable Parameters: {param_stats['trainable_percentage']:.2f}%")
print(f"Training Time: {training_time/60:.2f} minutes")
if torch.cuda.is_available():
    print(f"Peak GPU Memory: {peak_memory:.2f} GB")
print(f"Test Accuracy: {test_metrics['eval_accuracy']*100:.2f}%")
print(f"Test Macro F1: {test_metrics['eval_macro_f1']:.4f}")
print("="*70)