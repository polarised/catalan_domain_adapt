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
import torch.nn.utils.prune as prune
import numpy as np
import json
import time
import copy

# ========================================================================================
# ITERATIVE MAGNITUDE PRUNING FOR XLM-ROBERTA
# ========================================================================================
# This implements iterative pruning: prune → fine-tune → prune → fine-tune → ...
# Each round removes a percentage of lowest-magnitude weights across the model
# ========================================================================================

# -------------------------
# 0) Configuration
# -------------------------
ITERATIVE_CONFIG = {
    "num_rounds": 5,                    # Number of pruning rounds
    "prune_percent_per_round": 10,      # Remove 10% each round (total: 50%)
    "pruning_method": "global",         # "global" or "layer_wise"
    "fine_tune_epochs_per_round": 2,    # Fine-tune epochs after each pruning
    "pruning_type": "unstructured",     # "unstructured" (L1) or "structured"
}

# Target layers to prune (encoder layers only, not embeddings or classifier)
PRUNE_TARGET_MODULES = [
    "roberta.encoder.layer",  # All transformer encoder layers
]

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
BASE = os.path.join(script_dir, "Clean_Datasets")

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
print("\nBuilding POS tag vocabulary...")
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
# 4) Iterative Pruning Functions
# -------------------------

def get_prunable_parameters(model):
    """
    Get all parameters that should be pruned.
    Returns list of (module, parameter_name) tuples.
    """
    prunable_params = []
    
    for name, module in model.named_modules():
        # Only prune encoder layers (not embeddings or classifier)
        if any(target in name for target in PRUNE_TARGET_MODULES):
            # Prune weights in Linear and attention layers
            if isinstance(module, nn.Linear):
                prunable_params.append((module, 'weight'))
    
    return prunable_params


def calculate_sparsity(model):
    """
    Calculate current sparsity (percentage of zero weights) in the model.
    """
    total_params = 0
    zero_params = 0
    
    for name, module in model.named_modules():
        if any(target in name for target in PRUNE_TARGET_MODULES):
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask'):
                    # If pruned, use mask
                    weight = module.weight * module.weight_mask
                else:
                    weight = module.weight
                
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
    
    sparsity = (zero_params / total_params * 100) if total_params > 0 else 0
    return sparsity, total_params, zero_params

def get_incremental_prune_amount(round_idx, total_rounds, per_round):
    """
    Converts per-round pruning into correct incremental global pruning.
    """
    target = per_round * round_idx
    prev = per_round * (round_idx - 1)
    return (target - prev) / (1 - prev)


def apply_global_pruning(model, amount):
    """
    Apply global magnitude pruning across all parameters.
    
    Args:
        model: The model to prune
        amount: Fraction of parameters to prune (0.0 to 1.0)
    """
    print(f"\n{'='*70}")
    print(f"Applying global magnitude pruning: {amount*100:.1f}%")
    print(f"{'='*70}")
    
    prunable_params = get_prunable_parameters(model)
    
    # Apply L1 unstructured pruning globally
    prune.global_unstructured(
        prunable_params,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    sparsity, total, zeros = calculate_sparsity(model)
    print(f"Current sparsity: {sparsity:.2f}% ({zeros:,} / {total:,} parameters)")
    
    return model


def make_pruning_permanent(model):
    """
    Make pruning permanent by removing the mask and zeroing weights.
    This is called at the end to finalize the pruned model.
    """
    print("\nMaking pruning permanent...")
    
    for name, module in model.named_modules():
        if any(target in name for target in PRUNE_TARGET_MODULES):
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_orig'):
                    # Remove the pruning reparametrization
                    prune.remove(module, 'weight')
    
    print("✓ Pruning finalized")
    return model


def count_parameters(model):
    """Count total and non-zero parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count non-zero parameters
    nonzero_params = 0
    for p in model.parameters():
        nonzero_params += (p != 0).sum().item()
    
    sparsity = (1 - nonzero_params / total_params) * 100
    
    return {
        "total": total_params,
        "nonzero": nonzero_params,
        "sparsity": sparsity
    }

# -------------------------
# 5) Metrics computation
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
    
    # Calculate accuracy
    correct = sum(p == l for preds, labs in zip(true_predictions, true_labels) 
                  for p, l in zip(preds, labs))
    total = sum(len(labs) for labs in true_labels)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate macro F1
    tag_stats = {tag: {"tp": 0, "fp": 0, "fn": 0} for tag in UPOS}
    
    for preds, labs in zip(true_predictions, true_labels):
        for p, l in zip(preds, labs):
            if p == l:
                tag_stats[l]["tp"] += 1
            else:
                tag_stats[p]["fp"] += 1
                tag_stats[l]["fn"] += 1
    
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
# 6) Load model and prepare datasets
# -------------------------
print("\n" + "="*70)
print("ITERATIVE MAGNITUDE PRUNING")
print("="*70)
print(f"Configuration:")
print(f"  - Number of rounds: {ITERATIVE_CONFIG['num_rounds']}")
print(f"  - Prune per round: {ITERATIVE_CONFIG['prune_percent_per_round']}%")
print(f"  - Total target sparsity: {ITERATIVE_CONFIG['num_rounds'] * ITERATIVE_CONFIG['prune_percent_per_round']}%")
print(f"  - Fine-tune epochs per round: {ITERATIVE_CONFIG['fine_tune_epochs_per_round']}")
print("="*70)

print("\nLoading tokenizer and model...")
    
FINETUNED_MODEL_PATH = "/fhome/amlai09/catalan_domain_adapt/Training_Baseline/model_output"
tokenizer = XLMRobertaTokenizerFast.from_pretrained(FINETUNED_MODEL_PATH)
model = XLMRobertaForTokenClassification.from_pretrained(
    FINETUNED_MODEL_PATH
).to(device)

print("Loaded model num_labels:", model.config.num_labels)
print("Loaded id2label:", model.config.id2label)

if model.config.num_labels == len(UPOS):
    model.config.id2label = id2upos
    model.config.label2id = upos2id
else:
    print("⚠️ Warning: label mismatch, keeping model config labels")

# Count initial parameters
initial_params = count_parameters(model)
print(f"\nInitial parameters: {initial_params['total']:,}")

# Convert datasets
print("\nConverting datasets...")
train_ds = to_hf_dataset(train_data, tokenizer, upos2id)
dev_ds   = to_hf_dataset(dev_data, tokenizer, upos2id)
test_ds  = to_hf_dataset(test_data, tokenizer, upos2id)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

# -------------------------
# 7) Iterative Pruning Loop
# -------------------------
batch_size = 16 if torch.cuda.is_available() else 8

# Store results for each round
round_results = []
total_training_time = 0

print("\n" + "="*70)
print("STARTING ITERATIVE PRUNING")
print("="*70 + "\n")

# Initial evaluation (Round 0 - before any pruning)
print("="*70)
print("ROUND 0: BASELINE (No Pruning)")
print("="*70)

# Create initial trainer for baseline evaluation
baseline_args = TrainingArguments(
    output_dir="./Training/iterative_pruning_temp",
    per_device_eval_batch_size=batch_size * 2,
    dataloader_num_workers=0,
    fp16=torch.cuda.is_available(),
)

baseline_trainer = Trainer(
    model=model,
    args=baseline_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

baseline_metrics = baseline_trainer.evaluate(eval_dataset=test_ds)
params_info = count_parameters(model)

round_results.append({
    "round": 0,
    "sparsity": params_info['sparsity'],
    "nonzero_params": params_info['nonzero'],
    "test_accuracy": baseline_metrics['eval_accuracy'],
    "test_macro_f1": baseline_metrics['eval_macro_f1'],
    "test_loss": baseline_metrics['eval_loss'],
    "training_time": 0,
})

print(f"\nBaseline Results:")
print(f"  Test Accuracy: {baseline_metrics['eval_accuracy']:.4f} ({baseline_metrics['eval_accuracy']*100:.2f}%)")
print(f"  Macro F1: {baseline_metrics['eval_macro_f1']:.4f}")
print(f"  Sparsity: {params_info['sparsity']:.2f}%")

# Iterative pruning rounds
for round_num in range(1, ITERATIVE_CONFIG['num_rounds'] + 1):
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")

    print("\n" + "="*70)
    print(f"ROUND {round_num}: PRUNE + FINE-TUNE")
    print("="*70)
    
    round_start = time.time()
    
    # Step 1: Apply pruning
    prune_amount = get_incremental_prune_amount(
        round_num,
        ITERATIVE_CONFIG["num_rounds"],
        ITERATIVE_CONFIG["prune_percent_per_round"] / 100.0
    )
    model = apply_global_pruning(model, prune_amount)
    
    # Step 2: Fine-tune to recover from pruning
    print(f"\nFine-tuning for {ITERATIVE_CONFIG['fine_tune_epochs_per_round']} epochs...")
    
    training_args = TrainingArguments(
        output_dir=f"./Training/iterative_pruning_round_{round_num}",
        eval_strategy="epoch",
        save_strategy="no",  # Don't save checkpoints during iterative rounds
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=ITERATIVE_CONFIG['fine_tune_epochs_per_round'],
        weight_decay=0.01,
        warmup_ratio=0.05,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
        disable_tqdm=False,
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
    
    # Fine-tune
    trainer.train()
    
    # Step 3: Evaluate on test set
    print("\nEvaluating after fine-tuning...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    
    # Calculate round statistics
    round_time = time.time() - round_start
    total_training_time += round_time
    params_info = count_parameters(model)
    
    # Store results
    round_results.append({
        "round": round_num,
        "sparsity": params_info['sparsity'],
        "nonzero_params": params_info['nonzero'],
        "test_accuracy": test_metrics['eval_accuracy'],
        "test_macro_f1": test_metrics['eval_macro_f1'],
        "test_loss": test_metrics['eval_loss'],
        "training_time": round_time / 60,
    })
    
    # Print round summary
    print(f"\n{'='*70}")
    print(f"ROUND {round_num} SUMMARY")
    print(f"{'='*70}")
    print(f"Sparsity: {params_info['sparsity']:.2f}%")
    print(f"Non-zero params: {params_info['nonzero']:,} / {params_info['total']:,}")
    print(f"Test Accuracy: {test_metrics['eval_accuracy']:.4f} ({test_metrics['eval_accuracy']*100:.2f}%)")
    print(f"Macro F1: {test_metrics['eval_macro_f1']:.4f}")
    print(f"Round time: {round_time/60:.2f} minutes")
    
    # Calculate accuracy drop from baseline
    acc_drop = (baseline_metrics['eval_accuracy'] - test_metrics['eval_accuracy']) * 100
    print(f"Accuracy drop from baseline: {acc_drop:.2f}%")
    print(f"{'='*70}")

# -------------------------
# 8) Finalize and save
# -------------------------
print("\n" + "="*70)
print("FINALIZING PRUNED MODEL")
print("="*70)

# Make pruning permanent
model = make_pruning_permanent(model)

# Final evaluation
final_trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./Training/iterative_pruning_final",
        per_device_eval_batch_size=batch_size * 2,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    ),
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

final_test_metrics = final_trainer.evaluate(eval_dataset=test_ds)
final_params = count_parameters(model)

print(f"\nFinal Model Statistics:")
print(f"  Sparsity: {final_params['sparsity']:.2f}%")
print(f"  Non-zero parameters: {final_params['nonzero']:,}")
print(f"  Test Accuracy: {final_test_metrics['eval_accuracy']:.4f} ({final_test_metrics['eval_accuracy']*100:.2f}%)")
print(f"  Macro F1: {final_test_metrics['eval_macro_f1']:.4f}")

# Save final model
output_dir = "./Training/iterative_pruning_final"
os.makedirs(output_dir, exist_ok=True)

final_trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Save comprehensive results
results_path = os.path.join(output_dir, "iterative_pruning_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({
        "config": ITERATIVE_CONFIG,
        "initial_params": initial_params,
        "final_params": final_params,
        "baseline_metrics": {
            "accuracy": float(baseline_metrics['eval_accuracy']),
            "macro_f1": float(baseline_metrics['eval_macro_f1']),
            "loss": float(baseline_metrics['eval_loss']),
        },
        "final_metrics": {
            "accuracy": float(final_test_metrics['eval_accuracy']),
            "macro_f1": float(final_test_metrics['eval_macro_f1']),
            "loss": float(final_test_metrics['eval_loss']),
        },
        "round_by_round": [
            {
                "round": r["round"],
                "sparsity_percent": float(r["sparsity"]),
                "nonzero_params": int(r["nonzero_params"]),
                "test_accuracy": float(r["test_accuracy"]),
                "test_macro_f1": float(r["test_macro_f1"]),
                "training_time_minutes": float(r["training_time"]),
            }
            for r in round_results
        ],
        "total_training_time_minutes": total_training_time / 60,
        "total_training_time_hours": total_training_time / 3600,
    }, f, ensure_ascii=False, indent=2)

# -------------------------
# 9) Create summary visualization
# -------------------------
print("\n" + "="*70)
print("CREATING RESULTS SUMMARY")
print("="*70)

# Print round-by-round table
print("\nRound-by-Round Results:")
print(f"{'Round':<8} {'Sparsity':<12} {'Test Acc':<12} {'Macro F1':<12} {'Acc Drop':<12}")
print("-" * 60)

baseline_acc = round_results[0]['test_accuracy']
for r in round_results:
    acc_drop = (baseline_acc - r['test_accuracy']) * 100
    print(f"{r['round']:<8} {r['sparsity']:>6.2f}%     {r['test_accuracy']*100:>6.2f}%     "
          f"{r['test_macro_f1']:>6.4f}     {acc_drop:>6.2f}%")

print("\n" + "="*70)
print("ITERATIVE PRUNING COMPLETE!")
print("="*70)
print(f"\nFinal Results:")
print(f"  Total Sparsity: {final_params['sparsity']:.2f}%")
print(f"  Parameters Removed: {initial_params['total'] - final_params['nonzero']:,}")
print(f"  Final Accuracy: {final_test_metrics['eval_accuracy']*100:.2f}%")
print(f"  Accuracy Drop: {(baseline_acc - final_test_metrics['eval_accuracy'])*100:.2f}%")
print(f"  Total Training Time: {total_training_time/60:.2f} minutes")
print(f"\n✓ Results saved to: {results_path}")
print("="*70)