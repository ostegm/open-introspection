"""Verify sparse feature reconstruction from phase0 2x2 pickle.

Prints top features for 2 specific trials so you can cross-check
in the notebook with:
    results_2x2[cond]["features"][layer][trial][token].topk(10)
"""

import pickle
import numpy as np
from pathlib import Path

PKL = Path.home() / "gdrive/open_introspection/phase0_2x2_sparse_4b_20260208_042043.pkl"

with open(PKL, "rb") as f:
    data = pickle.load(f)

print(f"Model: {data['model']}")
print(f"Concept: {data['concept']}")
print(f"Trials per condition: {data['n_trials']}")
print(f"SAE layers: {data['sae_layers']}")
print(f"Sparse threshold: {data['sparse_threshold']}")
print(f"Conditions: {data['conditions']}")
print()

# Verify 2 trials: condition A trial 0, condition D trial 0
CHECKS = [
    ("A", 0, 22, 0),  # cond, trial, layer, token_idx
    ("A", 0, 22, 4),
    ("D", 0, 22, 0),
    ("A", 5, 17, 0),  # L17 — check the metacognitive layer
    ("A", 5, 17, 4),
]

for cond, trial, layer, tok_idx in CHECKS:
    trial_tokens = data["tokens"][cond][trial]
    trial_sparse = data["sparse_features"][cond][layer][trial]
    n_tokens = len(trial_sparse)

    print(f"=== Condition {cond}, Trial {trial}, Layer {layer}, Token {tok_idx} ===")
    print(f"  Total tokens in trial: {n_tokens}")
    print(f"  Token ID at position {tok_idx}: {trial_tokens[tok_idx]}")

    sparse_tok = trial_sparse[tok_idx]
    indices = sparse_tok["indices"]
    values = sparse_tok["values"]
    print(f"  Active features (>{data['sparse_threshold']}): {len(indices)}")

    if len(indices) == 0:
        print("  (no active features)")
        print()
        continue

    # Top 10 by activation value
    top_order = np.argsort(values)[::-1][:10]
    print(f"  Top 10 features:")
    print(f"  {'Feature':>8}  {'Value':>8}")
    for i in top_order:
        print(f"  {indices[i]:>8}  {values[i]:>8.2f}")
    print()

# Summary stats
print("=" * 60)
print("SUMMARY STATS")
print("=" * 60)
for cond in sorted(data["sparse_features"]):
    for layer in data["sae_layers"]:
        trials = data["sparse_features"][cond][layer]
        n_trials = len(trials)
        total_active = sum(len(t["indices"]) for trial in trials for t in trial)
        total_tokens = sum(len(trial) for trial in trials)
        avg_active = total_active / total_tokens if total_tokens > 0 else 0
        print(f"  {cond} L{layer}: {n_trials} trials, {total_tokens} tokens, "
              f"{avg_active:.0f} avg active features/token")
