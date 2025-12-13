"""
Automated script to run all pruning experiments sequentially.
This will take several hours depending on your hardware.

Usage:
    python run_all_pruning_experiments.py
"""

import subprocess
import sys
import time
import os
import re

# ========================================================================================
# CONFIGURATION
# ========================================================================================

# All experiments to run
EXPERIMENTS = [
    # Baseline (must run first!)
    "baseline",
    
    # Layer pruning experiments
    "layer_keep_top_8",
    "layer_keep_top_6",
    "layer_keep_bottom_8",
    "layer_keep_bottom_6",
    "layer_keep_middle_8",
    
    # Head pruning experiments
    "head_prune_50",
    "head_prune_67",
    "head_prune_75",
]

# Script to modify and run
script_dir = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(script_dir, "pruning_training.py")

# ========================================================================================
# HELPER FUNCTIONS
# ========================================================================================

def modify_experiment_config(script_path, experiment_name):
    """
    Modify the CURRENT_EXPERIMENT variable in the script.
    
    Args:
        script_path: Path to pruning_experiments.py
        experiment_name: Name of experiment to set
    """
    print(f"\n📝 Updating {script_path} to run: {experiment_name}")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace the CURRENT_EXPERIMENT line
    pattern = r'CURRENT_EXPERIMENT\s*=\s*["\'][^"\']+["\']'
    replacement = f'CURRENT_EXPERIMENT = "{experiment_name}"'
    
    new_content = re.sub(pattern, replacement, content)
    
    with open(script_path, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Set CURRENT_EXPERIMENT = '{experiment_name}'")


def check_experiment_completed(experiment_name):
    """
    Check if an experiment has already been completed.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        bool: True if results exist, False otherwise
    """
    results_path = f"./Training/pruning_{experiment_name}/pruning_results.json"
    return os.path.exists(results_path)


def format_time(seconds):
    """Format seconds into hours:minutes:seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_header(text):
    """Print a formatted header."""
    print_separator()
    print(text.center(80))
    print_separator()

# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

def main():
    print_header("AUTOMATED PRUNING EXPERIMENTS")
    print(f"\n📊 Total experiments to run: {len(EXPERIMENTS)}")
    print(f"📝 Script: {SCRIPT_PATH}")
    print(f"\n🕐 Estimated total time: {len(EXPERIMENTS) * 30}-{len(EXPERIMENTS) * 60} minutes")
    print(f"   (30-60 minutes per experiment)")
    
    # Check which experiments are already completed
    completed = []
    pending = []
    
    for exp in EXPERIMENTS:
        if check_experiment_completed(exp):
            completed.append(exp)
        else:
            pending.append(exp)
    
    if completed:
        print(f"\n✓ Already completed: {len(completed)}")
        for exp in completed:
            print(f"    - {exp}")
    
    if pending:
        print(f"\n⏳ Pending: {len(pending)}")
        for exp in pending:
            print(f"    - {exp}")
    else:
        print("\n✓ All experiments already completed!")
        print("\nTo rerun, delete the output directories:")
        print("    rm -rf ./Training/pruning_*")
        return
    
    # Auto-start without confirmation
    print(f"\n{'='*80}")
    print(f"Starting {len(pending)} experiments automatically...")
    print(f"{'='*80}")
    
    # Run experiments
    total_start_time = time.time()
    results_summary = []
    
    print_header("STARTING EXPERIMENTS")
    
    for i, exp_name in enumerate(pending, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(pending)}: {exp_name}")
        print(f"{'='*80}")
        
        exp_start_time = time.time()
        
        try:
            # Modify the script
            modify_experiment_config(SCRIPT_PATH, exp_name)
            
            # Run the experiment
            print(f"\n🚀 Running experiment...")
            print(f"Command: python {SCRIPT_PATH}")
            print(f"{'='*80}\n")
            
            result = subprocess.run(
                [sys.executable, SCRIPT_PATH],
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            exp_time = time.time() - exp_start_time
            
            print(f"\n{'='*80}")
            print(f"✓ Experiment '{exp_name}' completed successfully")
            print(f"✓ Time: {format_time(exp_time)}")
            print(f"{'='*80}")
            
            results_summary.append({
                "experiment": exp_name,
                "status": "SUCCESS",
                "time": exp_time
            })
            
        except subprocess.CalledProcessError as e:
            exp_time = time.time() - exp_start_time
            
            print(f"\n{'='*80}")
            print(f"✗ Experiment '{exp_name}' FAILED")
            print(f"✗ Error code: {e.returncode}")
            print(f"✗ Time before failure: {format_time(exp_time)}")
            print(f"{'='*80}")
            
            results_summary.append({
                "experiment": exp_name,
                "status": "FAILED",
                "time": exp_time,
                "error": str(e)
            })
            
            # Auto-continue on failure
            print("\n⚠ Continuing with remaining experiments...")
            continue
        
        except KeyboardInterrupt:
            print("\n\n❌ Interrupted by user (Ctrl+C)")
            results_summary.append({
                "experiment": exp_name,
                "status": "INTERRUPTED",
                "time": time.time() - exp_start_time
            })
            break
    
    # ========================================================================================
    # FINAL SUMMARY
    # ========================================================================================
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*80)
    print_header("EXPERIMENTS COMPLETE")
    
    print(f"\n📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Experiments run: {len(results_summary)}/{len(pending)}")
    
    # Count by status
    success_count = sum(1 for r in results_summary if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results_summary if r["status"] == "FAILED")
    interrupted_count = sum(1 for r in results_summary if r["status"] == "INTERRUPTED")
    
    print(f"\n✓ Successful: {success_count}")
    print(f"✗ Failed: {failed_count}")
    if interrupted_count > 0:
        print(f"⏸ Interrupted: {interrupted_count}")
    
    # Detailed results
    print(f"\n{'='*80}")
    print("DETAILED RESULTS:")
    print(f"{'='*80}")
    print(f"{'Experiment':<25} {'Status':<15} {'Time':<15}")
    print(f"{'-'*80}")
    
    for result in results_summary:
        status_symbol = "✓" if result["status"] == "SUCCESS" else "✗"
        print(f"{result['experiment']:<25} {status_symbol} {result['status']:<13} {format_time(result['time']):<15}")
    
    # Failed experiments
    if failed_count > 0:
        print(f"\n{'='*80}")
        print("FAILED EXPERIMENTS (can be retried):")
        print(f"{'='*80}")
        for result in results_summary:
            if result["status"] == "FAILED":
                print(f"  - {result['experiment']}")
                if "error" in result:
                    print(f"    Error: {result['error']}")
    
    # Next steps
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    
    if success_count > 0:
        print("\n✓ Run comparison and analysis:")
        print("    python pruning_comparison.py")
    
    if failed_count > 0:
        print(f"\n⚠ Retry failed experiments:")
        print(f"    Edit this script to only run failed experiments")
    
    print(f"\n✓ Check results in:")
    print(f"    ./Training/pruning_*/")
    
    print(f"\n{'='*80}")
    print("ALL DONE! 🎉")
    print(f"{'='*80}\n")


# ========================================================================================
# RUN
# ========================================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"FATAL ERROR")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)