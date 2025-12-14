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
import torch.nn as nn
import numpy as np
import json
import time
import copy

# ========================================================================================
# PRUNING EXPERIMENTS FOR XLM-ROBERTA ON CATALAN POS TAGGING
# ========================================================================================
# This script implements structured pruning (layer pruning and head pruning)
# Based on the research roadmap for model compression
# ========================================================================================

# -------------------------
# 0) Configuration
# -------------------------
PRUNING_EXPERIMENTS = {
    # ===== LAYER PRUNING EXPERIMENTS =====
    "baseline": {
        "type": "none",
        "description": "Baseline (no pruning)",
        "config": None
    },
    
    # Keep top layers (remove bottom layers)
    "layer_keep_top_8": {
        "type": "layer",
        "description": "Keep top 8 layers (remove bottom 4)",
        "config": {"keep_layers": list(range(4, 12))}  # Keep layers 4-11
    },
    "layer_keep_top_6": {
        "type": "layer",
        "description": "Keep top 6 layers (remove bottom 6)",
        "config": {"keep_layers": list(range(6, 12))}  # Keep layers 6-11
    },
    
    # Keep bottom layers (remove top layers)
    "layer_keep_bottom_8": {
        "type": "layer",
        "description": "Keep bottom 8 layers (remove top 4)",
        "config": {"keep_layers": list(range(0, 8))}  # Keep layers 0-7
    },
    "layer_keep_bottom_6": {
        "type": "layer",
        "description": "Keep bottom 6 layers (remove top 6)",
        "config": {"keep_layers": list(range(0, 6))}  # Keep layers 0-5
    },
    
    # Keep middle layers
    "layer_keep_middle_8": {
        "type": "layer",
        "description": "Keep middle 8 layers",
        "config": {"keep_layers": list(range(2, 10))}  # Keep layers 2-9
    },
    
    # ===== HEAD PRUNING EXPERIMENTS =====
    "head_prune_50": {
        "type": "head",
        "description": "Prune 50% of attention heads (6/12 per layer)",
        "config": {"heads_to_keep": 6}  # Keep 6 out of 12 heads per layer
    },
    "head_prune_67": {
        "type": "head",
        "description": "Prune 67% of attention heads (4/12 per layer)",
        "config": {"heads_to_keep": 4}  # Keep 4 out of 12 heads per layer
    },
    "head_prune_75": {
        "type": "head",
        "description": "Prune 75% of attention heads (3/12 per layer)",
        "config": {"heads_to_keep": 3}  # Keep 3 out of 12 heads per layer
    },
}

# Set which experiment to run
CURRENT_EXPERIMENT = "baseline"  # Change this to run different experiments

# -------------------------
# 1) GPU setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# -------------------------
# 2) Load datasets
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

# -------------------------
# 4) Pruning Functions
# -------------------------

def prune_layers(model, keep_layers):
    """
    Prune transformer layers by removing specified layers.
    
    Args:
        model: XLMRobertaForTokenClassification model
        keep_layers: List of layer indices to keep (e.g., [0, 1, 2, 7, 8, 9, 10, 11])
    """
    print(f"\nPruning layers... Keeping layers: {keep_layers}")
    
    # Get the encoder layers
    encoder = model.roberta.encoder
    all_layers = list(encoder.layer)
    
    # Keep only specified layers
    pruned_layers = nn.ModuleList([all_layers[i] for i in keep_layers])
    encoder.layer = pruned_layers
    
    # Update config
    model.config.num_hidden_layers = len(keep_layers)
    
    print(f"Reduced from 12 to {len(keep_layers)} layers")
    return model


def calculate_head_importance(model, dataloader, device):
    """Calculate importance of each attention head based on attention weights."""
    print("\nCalculating head importance scores...")
    
    # IMPORTANT: Ensure model is on correct device and in eval mode
    model = model.to(device)  # Add this line
    model.eval()
    
    # Initialize importance scores
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_importance = {(layer, head): 0.0 for layer in range(num_layers) for head in range(num_heads)}
    
    num_batches = min(50, len(dataloader))  # Sample first 50 batches for speed
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            
            # Forward pass with attention outputs
            outputs = model.roberta(**inputs, output_attentions=True)
            attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)
            
            # Calculate importance as L2 norm of attention weights
            for layer_idx, layer_attn in enumerate(attentions):
                # layer_attn shape: (batch, num_heads, seq_len, seq_len)
                for head_idx in range(num_heads):
                    head_attn = layer_attn[:, head_idx, :, :]  # (batch, seq, seq)
                    importance = head_attn.norm(p=2).item()
                    head_importance[(layer_idx, head_idx)] += importance
    
    # Average importance across batches
    for key in head_importance:
        head_importance[key] /= num_batches
    
    return head_importance


def prune_heads(model, heads_to_keep_per_layer, train_dataloader=None):
    """Prune attention heads based on importance."""
    print(f"\nPruning heads... Keeping {heads_to_keep_per_layer} heads per layer")
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    if train_dataloader is not None:
        # Make sure model is on device before calculating importance
        current_device = next(model.parameters()).device  # Get model's device
        head_importance = calculate_head_importance(model, train_dataloader, current_device)
        
        # For each layer, keep top-k most important heads
        heads_to_prune = {}
        for layer_idx in range(num_layers):
            layer_scores = [(head, head_importance[(layer_idx, head)]) 
                           for head in range(num_heads)]
            layer_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Prune least important heads
            heads_to_prune_in_layer = [head for head, _ in layer_scores[heads_to_keep_per_layer:]]
            heads_to_prune[layer_idx] = heads_to_prune_in_layer
            
            print(f"  Layer {layer_idx}: Keeping heads {[h for h, _ in layer_scores[:heads_to_keep_per_layer]]}")
    else:
        # Simple strategy: keep first k heads
        heads_to_prune = {
            layer: list(range(heads_to_keep_per_layer, num_heads))
            for layer in range(num_layers)
        }
        print(f"  Using simple strategy: keeping first {heads_to_keep_per_layer} heads per layer")
    
    # Apply pruning
    model.roberta.prune_heads(heads_to_prune)
    
    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "percentage_remaining": 100.0
    }


def apply_pruning(model, experiment_config, train_dataloader=None):
    """
    Apply pruning based on experiment configuration.
    
    Args:
        model: Base model to prune
        experiment_config: Dict with pruning configuration
        train_dataloader: DataLoader for head importance calculation
        
    Returns:
        pruned_model: Model after pruning
        pruning_stats: Statistics about the pruning
    """
    exp_type = experiment_config["type"]
    config = experiment_config["config"]
    
    if exp_type == "none":
        # No pruning (baseline)
        return model, {"pruning_type": "none"}
    
    elif exp_type == "layer":
        # Layer pruning
        keep_layers = config["keep_layers"]
        original_layers = 12
        model = prune_layers(model, keep_layers)
        
        return model, {
            "pruning_type": "layer",
            "layers_kept": len(keep_layers),
            "layers_removed": original_layers - len(keep_layers),
            "percentage_layers_remaining": (len(keep_layers) / original_layers) * 100
        }
    
    elif exp_type == "head":
        # Head pruning
        heads_to_keep = config["heads_to_keep"]
        original_heads = 12
        model = prune_heads(model, heads_to_keep, train_dataloader)
        
        return model, {
            "pruning_type": "head",
            "heads_per_layer_kept": heads_to_keep,
            "heads_per_layer_removed": original_heads - heads_to_keep,
            "percentage_heads_remaining": (heads_to_keep / original_heads) * 100
        }
    
    else:
        raise ValueError(f"Unknown pruning type: {exp_type}")

# -------------------------
# 5) Load and configure model
# -------------------------
print("\n" + "="*70)
print(f"EXPERIMENT: {PRUNING_EXPERIMENTS[CURRENT_EXPERIMENT]['description']}")
print("="*70)

print("\nLoading tokenizer and base model...")
FINETUNED_MODEL_PATH = "/fhome/amlai09/catalan_domain_adapt/Training_Baseline/model_output"
tokenizer = XLMRobertaTokenizerFast.from_pretrained(FINETUNED_MODEL_PATH)
model = XLMRobertaForTokenClassification.from_pretrained(
    FINETUNED_MODEL_PATH
).to(device)

# Add label mappings
model.config.id2label = id2upos
model.config.label2id = upos2id

# Count parameters before pruning
params_before = count_parameters(model)
print(f"\nParameters before pruning: {params_before['total']:,}")

# -------------------------
# 6) Convert datasets
# -------------------------
print("\nConverting datasets to HuggingFace format...")
train_ds = to_hf_dataset(train_data, tokenizer, upos2id)
dev_ds   = to_hf_dataset(dev_data, tokenizer, upos2id)
test_ds  = to_hf_dataset(test_data, tokenizer, upos2id)

print(f"Train dataset: {len(train_ds)} samples")
print(f"Dev dataset: {len(dev_ds)} samples")
print(f"Test dataset: {len(test_ds)} samples")

# -------------------------
# 7) Apply pruning
# -------------------------
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

# Create a small dataloader for head importance calculation
if PRUNING_EXPERIMENTS[CURRENT_EXPERIMENT]["type"] == "head":
    from torch.utils.data import DataLoader
    importance_loader = DataLoader(
        train_ds, 
        batch_size=8, 
        collate_fn=data_collator,
        shuffle=False
    )
else:
    importance_loader = None

# Apply pruning - keep model on device
model = model.to(device)  # ← Move model FIRST
model, pruning_stats = apply_pruning(
    model, 
    PRUNING_EXPERIMENTS[CURRENT_EXPERIMENT],
    importance_loader
)

# Count parameters after pruning
params_after = count_parameters(model)
param_reduction = ((params_before['total'] - params_after['total']) / params_before['total']) * 100

print(f"\n{'='*70}")
print("PRUNING SUMMARY")
print(f"{'='*70}")
print(f"Parameters before: {params_before['total']:,}")
print(f"Parameters after:  {params_after['total']:,}")
print(f"Reduction: {param_reduction:.2f}%")
print(f"{'='*70}")

# -------------------------
# 8) Metrics computation
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
    
    # Calculate per-tag metrics
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
    for tag, stats in tag_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores) if f1_scores else 0
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

# -------------------------
# 9) Training configuration
# -------------------------
batch_size = 16 if torch.cuda.is_available() else 8

output_dir = f"./Training/pruning_{CURRENT_EXPERIMENT}"
logging_dir = f"./Training/logs_pruning_{CURRENT_EXPERIMENT}"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
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
    num_train_epochs=3,  # Reduced for pruning experiments
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
# 10) Trainer
# -------------------------
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
# 11) Train
# -------------------------
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Training samples: {len(train_ds)}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {training_args.num_train_epochs}")
print("="*70 + "\n")

# Measure GPU memory
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

start_time = time.time()

try:
    trainer.train()
    training_time = time.time() - start_time
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n✓ Training completed in {training_time/60:.2f} minutes!")
        print(f"✓ Peak GPU memory: {peak_memory:.2f} GB")
    else:
        print(f"\n✓ Training completed in {training_time/60:.2f} minutes!")
        peak_memory = 0
        
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    raise

# -------------------------
# 12) Evaluate
# -------------------------
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

dev_metrics = trainer.evaluate(eval_dataset=dev_ds)
test_metrics = trainer.evaluate(eval_dataset=test_ds)

print("\nTest Set Results:")
print(f"  Accuracy: {test_metrics['eval_accuracy']:.4f} ({test_metrics['eval_accuracy']*100:.2f}%)")
print(f"  Macro F1: {test_metrics['eval_macro_f1']:.4f}")
print(f"  Loss: {test_metrics['eval_loss']:.4f}")

# -------------------------
# 13) Per-tag analysis
# -------------------------
print("\n" + "="*70)
print("PER-TAG PERFORMANCE")
print("="*70)

predictions = trainer.predict(test_ds)
pred_labels = np.argmax(predictions.predictions, axis=2)

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
# 14) Save results
# -------------------------
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Save comprehensive results
results_path = os.path.join(output_dir, "pruning_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({
        "experiment_name": CURRENT_EXPERIMENT,
        "experiment_config": PRUNING_EXPERIMENTS[CURRENT_EXPERIMENT],
        "pruning_stats": pruning_stats,
        "parameters_before": params_before['total'],
        "parameters_after": params_after['total'],
        "parameter_reduction_percent": param_reduction,
        "training_time_minutes": training_time / 60,
        "peak_gpu_memory_gb": peak_memory if torch.cuda.is_available() else None,
        "test_metrics": {
            "accuracy": float(test_metrics['eval_accuracy']),
            "macro_f1": float(test_metrics['eval_macro_f1']),
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
        }
    }, f, ensure_ascii=False, indent=2)

print(f"\n✓ Results saved to: {results_path}")

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print(f"Experiment: {PRUNING_EXPERIMENTS[CURRENT_EXPERIMENT]['description']}")
print(f"Parameter Reduction: {param_reduction:.2f}%")
print(f"Test Accuracy: {test_metrics['eval_accuracy']*100:.2f}%")
print(f"Training Time: {training_time/60:.2f} minutes")
print("="*70)