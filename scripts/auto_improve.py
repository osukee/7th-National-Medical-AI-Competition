#!/usr/bin/env python3
"""
Auto-Improvement Engine for Kaggle Competition.

Reads the current LB score + experiment history, applies deterministic
config improvements, and modifies train_notebook.py Config class.

Usage:
    python scripts/auto_improve.py --score 0.44102 --config kaggle/train_notebook.py

Stop conditions:
    - Score >= target (default 0.47)
    - Max iterations reached (default 5)
    - No improvement for 2 consecutive runs
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime


STATE_FILE = Path("experiments/auto_iterations.json")
CONFIG_FILE = Path("kaggle/train_notebook.py")

# Ordered list of improvements to try, one per iteration
IMPROVEMENT_QUEUE = [
    {
        "name": "resolution_576",
        "description": "Increase resolution to 576 for better SSIM structure detail",
        "changes": {
            "image_size": "576",
            "batch_size": "2",
            "learning_rate": "8e-5",
            "gradient_checkpointing": "True",
        },
    },
    {
        "name": "encoder_b6_scse",
        "description": "Upgrade to EfficientNet-b6 with scSE decoder attention",
        "changes": {
            "encoder": '"efficientnet-b6"',
            "decoder_attention_type": '"scse"',
            "batch_size": "2",
            "learning_rate": "7e-5",
        },
    },
    {
        "name": "cosine_24ep",
        "description": "Extend to 24 epochs with lower LR and weight decay for convergence",
        "changes": {
            "epochs": "24",
            "learning_rate": "7e-5",
            "weight_decay": "5e-6",
        },
    },
    {
        "name": "pseudo_label_strengthen",
        "description": "Strengthen pseudo-labeling with more epochs and higher weight",
        "changes": {
            "pseudo_label_epochs": "15",
            "pseudo_label_weight": "0.7",
            "pseudo_label_lr_factor": "0.2",
            "tta_aggregate": '"mean"',
        },
    },
    {
        "name": "inference_tune",
        "description": "Inference tuning: top-2 fold ensemble + CLAHE preprocessing",
        "changes": {
            "n_folds_ensemble": "2",
            "fold_rank_weights": "[1.0, 0.8]",
            "tta_aggregate": '"mean"',
            "clahe_clip_limit": "3.0",
        },
    },
]

# Known-good config from Phase 1 final state (score 0.44049)
KNOWN_GOOD_CONFIG = {
    "encoder": '"efficientnet-b5"',
    "batch_size": "4",
    "epochs": "20",
    "loss_type": '"optimized"',
    "tta_mode": '"dihedral8"',
    "n_folds_ensemble": "3",
    "fold_rank_weights": "[1.0, 0.7, 0.4]",
    "pseudo_label_enabled": "True",
    "gradient_checkpointing": "True",
    "augmentation_strength": "0.7",
    "image_size": "512",
    "decoder_attention_type": "None",
}


def load_state():
    """Load iteration state from JSON file with corruption handling."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Corrupt state file, resetting: {e}")
    return {
        "iteration": 0,
        "max_iterations": 5,
        "target_score": 0.47,
        "best_score": 0.0,
        "history": [],
        "consecutive_no_improvement": 0,
        "next_improvement_idx": 0,
    }


def save_state(state):
    """Save iteration state to JSON file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _find_config_class_range(content):
    """Find the line range of 'class Config:' block.

    Returns (start_idx, end_idx) as 0-based line indices (inclusive).
    The end is the last line before the next top-level class/function/block.
    """
    lines = content.split("\n")
    start = None
    end = None
    for i, line in enumerate(lines):
        if re.match(r"^class Config\b", line):
            start = i
            continue
        if start is not None and i > start:
            # Next top-level definition ends Config block
            if re.match(r"^(class |def |if |# ---)", line):
                end = i - 1
                break
    if start is not None and end is None:
        end = len(lines) - 1
    return start, end


def update_config(config_path, changes):
    """Update Config class attributes in train_notebook.py.

    SCOPED: Only replaces within the Config class block to avoid
    accidentally mutating function-level variables with same names.
    """
    content = config_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    start, end = _find_config_class_range(content)
    if start is None:
        print("  ERROR: Could not find 'class Config' block!")
        return

    print(f"  Config class found at lines {start+1}-{end+1}")

    for key, value in changes.items():
        found = False
        for i in range(start, end + 1):
            # Match: "    key = old_value  # optional comment"
            pattern = rf"^(    {key}\s*=\s*)([^\n#]+)(#.*)?$"
            match = re.match(pattern, lines[i])
            if match:
                lines[i] = f"    {key} = {value}  # auto-improved"
                print(f"  Updated L{i+1}: {key} = {value}")
                found = True
                break
        if not found:
            print(f"  Warning: '{key}' not found in Config block")

    config_path.write_text("\n".join(lines), encoding="utf-8")


def decide_improvement(state, current_score):
    """Decide what improvement to apply based on score and history."""
    best_score = state["best_score"]
    idx = state["next_improvement_idx"]

    # Score dropped significantly -> revert to known-good
    if best_score > 0 and current_score < best_score - 0.005:
        print(f"!! Score dropped ({current_score} < {best_score} - 0.005)")
        print("Reverting to known-good config")
        state["next_improvement_idx"] = idx + 1
        return "revert_known_good", "Revert to known-good config (score dropped)", KNOWN_GOOD_CONFIG

    # All improvements exhausted
    if idx >= len(IMPROVEMENT_QUEUE):
        print("All improvements exhausted")
        return None, None, None

    improvement = IMPROVEMENT_QUEUE[idx]
    state["next_improvement_idx"] = idx + 1
    return improvement["name"], improvement["description"], improvement["changes"]


def check_stop_conditions(state, current_score):
    """Check if the loop should stop."""
    reasons = []

    if current_score >= state["target_score"]:
        reasons.append(f"Target score {state['target_score']} reached! ({current_score})")

    # Off-by-one fix: use > instead of >= so iteration 5 can still apply
    if state["iteration"] > state["max_iterations"]:
        reasons.append(f"Max iterations ({state['max_iterations']}) exceeded")

    if state["consecutive_no_improvement"] >= 2:
        reasons.append("No improvement for 2 consecutive runs")

    if state["next_improvement_idx"] >= len(IMPROVEMENT_QUEUE):
        reasons.append("All improvements exhausted")

    return reasons


def main():
    parser = argparse.ArgumentParser(description="Auto-improve Kaggle config")
    parser.add_argument("--score", type=float, required=True, help="Current LB score")
    parser.add_argument("--config", type=str, default=str(CONFIG_FILE), help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files")
    args = parser.parse_args()

    config_path = Path(args.config)
    current_score = args.score

    print(f"\n{'='*60}")
    print(f"Auto-Improvement Engine")
    print(f"{'='*60}")
    print(f"Current Score: {current_score}")

    # Load state
    state = load_state()
    state["iteration"] += 1
    print(f"Iteration: {state['iteration']}/{state['max_iterations']}")
    print(f"Best Score: {state['best_score']}")

    # Record history
    state["history"].append({
        "iteration": state["iteration"],
        "score": current_score,
        "timestamp": datetime.now().isoformat(),
    })

    # Update best score tracking
    if current_score > state["best_score"]:
        print(f"** New best score! {state['best_score']} -> {current_score}")
        state["best_score"] = current_score
        state["consecutive_no_improvement"] = 0
    else:
        state["consecutive_no_improvement"] += 1
        print(f"No improvement ({state['consecutive_no_improvement']} consecutive)")

    # Check stop conditions
    stop_reasons = check_stop_conditions(state, current_score)
    if stop_reasons:
        print(f"\nSTOPPING: {'; '.join(stop_reasons)}")
        save_state(state)
        print("STOP")
        sys.exit(0)

    # Decide improvement
    name, description, changes = decide_improvement(state, current_score)

    if name is None:
        print("\nNo more improvements to try")
        save_state(state)
        print("STOP")
        sys.exit(0)

    print(f"\n>> Applying: {name}")
    print(f"   {description}")

    # Record what we're applying
    state["history"][-1]["improvement"] = name
    state["history"][-1]["description"] = description

    if not args.dry_run:
        update_config(config_path, changes)
        save_state(state)
        # Write experiment log
        exp_dir = Path(f"experiments/auto_{name}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        log = f"""# Auto-Improvement: {name}

> Iteration {state['iteration']} | Previous Score: {current_score} | {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Change
{description}

## Config Changes
{json.dumps(changes, indent=2)}
"""
        (exp_dir / "experiment_log.md").write_text(log, encoding="utf-8")
        print(f"\nChanges applied. Ready to commit and push.")
        print("CONTINUE")
    else:
        print("\n[DRY RUN] No files modified")
        save_state(state)
        print("CONTINUE")


if __name__ == "__main__":
    main()
