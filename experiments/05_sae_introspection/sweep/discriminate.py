#!/usr/bin/env python
"""Phase 3: Feature discrimination and clustering.

Loads judged sweep data, assigns contrast groups, computes per-feature
effect sizes (Cohen's d), and clusters correlated features.

Usage:
    uv run python experiments/05_sae_introspection/sweep/discriminate.py \\
        data/sweeps/sae-sweep-20260210/gemma-4b \\
        --output results/discrimination.json

    # Only A vs C contrast
    uv run python experiments/05_sae_introspection/sweep/discriminate.py \\
        data/sweeps/sae-sweep-20260210/gemma-4b \\
        --contrast A_vs_C
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from numpy.typing import NDArray

SAE_WIDTH = 262144  # 262k features


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_trials(sweep_dir: Path) -> list[dict[str, Any]]:
    """Load all judged trials from JSONL files."""
    trials = []
    for path in sorted(sweep_dir.rglob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trials.append(json.loads(line))
    return trials


def assign_groups(
    trials: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Assign trials to contrast groups based on judge labels.

    Groups:
        A: was_injected=True, judge.answer="pass" (injection detected)
        B: was_injected=False, judge.answer="pass" (correct no-detection)
        C: was_injected=True, judge.answer="fail", not refused (injection missed)
        excluded: refused or no judge result
    """
    groups: dict[str, list[dict[str, Any]]] = {
        "A": [], "B": [], "C": [], "excluded": [],
    }

    for trial in trials:
        judge = trial.get("judge")
        if not judge:
            groups["excluded"].append(trial)
            continue

        if judge.get("refused", False):
            groups["excluded"].append(trial)
            continue

        was_injected = trial["was_injected"]
        answer = judge["answer"]

        if was_injected and answer == "pass":
            groups["A"].append(trial)
        elif not was_injected and answer == "pass":
            groups["B"].append(trial)
        elif was_injected and answer == "fail":
            groups["C"].append(trial)
        else:
            # Control trial that failed (false positive) — exclude
            groups["excluded"].append(trial)

    return groups


# ── Feature Aggregation ──────────────────────────────────────────────────────


def aggregate_trial_features(
    trial: dict[str, Any],
    n_features: int = SAE_WIDTH,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute mean and max activation per feature across generated tokens.

    Args:
        trial: Trial record with sae_features field.
        n_features: Total SAE width.

    Returns:
        (mean_activations, max_activations) each of shape (n_features,)
    """
    sae_features = trial.get("sae_features", [])
    if not sae_features:
        return np.zeros(n_features, dtype=np.float32), np.zeros(n_features, dtype=np.float32)

    n_tokens = len(sae_features)
    # Accumulate sum and max
    feat_sum = np.zeros(n_features, dtype=np.float64)
    feat_max = np.zeros(n_features, dtype=np.float32)

    for token_feats in sae_features:
        indices = token_feats["indices"]
        values = token_feats["values"]
        for idx, val in zip(indices, values, strict=True):
            feat_sum[idx] += val
            if val > feat_max[idx]:
                feat_max[idx] = val

    feat_mean = (feat_sum / n_tokens).astype(np.float32)
    return feat_mean, feat_max


def build_feature_matrix(
    trials: list[dict[str, Any]],
    n_features: int = SAE_WIDTH,
    mode: str = "mean",
) -> NDArray[np.float32]:
    """Build (n_trials, n_features) matrix of aggregated activations.

    Args:
        trials: List of trial records.
        n_features: SAE width.
        mode: "mean" or "max" aggregation per trial.

    Returns:
        Array of shape (n_trials, n_features).
    """
    rows = []
    for trial in trials:
        mean_act, max_act = aggregate_trial_features(trial, n_features)
        rows.append(mean_act if mode == "mean" else max_act)
    return np.stack(rows)


# ── Discrimination ───────────────────────────────────────────────────────────


def cohens_d(
    group_a: NDArray[np.float32],
    group_b: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Compute Cohen's d for each feature between two groups.

    Args:
        group_a: Shape (n_a, n_features).
        group_b: Shape (n_b, n_features).

    Returns:
        Array of shape (n_features,) with effect sizes.
    """
    n_a, n_b = group_a.shape[0], group_b.shape[0]
    mean_a = group_a.mean(axis=0)
    mean_b = group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1)
    var_b = group_b.var(axis=0, ddof=1)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)

    # Avoid division by zero
    pooled_std = np.where(pooled_std > 0, pooled_std, 1.0)

    return (mean_a - mean_b) / pooled_std


def compute_contrast(
    group_a_trials: list[dict[str, Any]],
    group_b_trials: list[dict[str, Any]],
    n_features: int = SAE_WIDTH,
    mode: str = "mean",
    top_n: int = 500,
) -> list[dict[str, Any]]:
    """Compute feature discrimination between two groups.

    Returns top_n features ranked by absolute effect size.
    """
    mat_a = build_feature_matrix(group_a_trials, n_features, mode)
    mat_b = build_feature_matrix(group_b_trials, n_features, mode)

    d = cohens_d(mat_a, mat_b)
    mean_a = mat_a.mean(axis=0)
    mean_b = mat_b.mean(axis=0)

    # Rank by absolute effect size
    top_indices = np.argsort(np.abs(d))[::-1][:top_n]

    results = []
    for idx in top_indices:
        idx = int(idx)
        results.append({
            "feature": idx,
            "effect_size": float(d[idx]),
            "abs_effect_size": float(abs(d[idx])),
            "mean_A": float(mean_a[idx]),
            "mean_B": float(mean_b[idx]),
        })

    return results


# ── Clustering ───────────────────────────────────────────────────────────────


def cluster_features(
    trials: list[dict[str, Any]],
    feature_indices: list[int],
    n_features: int = SAE_WIDTH,
    mode: str = "mean",
    threshold: float = 0.5,
) -> list[list[int]]:
    """Cluster features by correlation across trials.

    Uses hierarchical clustering with distance = 1 - |r|.

    Args:
        trials: All trials used for correlation computation (A + B or A + C).
        feature_indices: Which features to cluster.
        n_features: SAE width.
        mode: Aggregation mode.
        threshold: Clustering distance threshold.

    Returns:
        List of clusters, each a list of feature indices.
    """
    if len(feature_indices) < 2:
        return [feature_indices] if feature_indices else []

    mat = build_feature_matrix(trials, n_features, mode)
    # Extract only the features we care about
    sub = mat[:, feature_indices]  # (n_trials, n_selected)

    # Pearson correlation
    corr = np.corrcoef(sub.T)  # (n_selected, n_selected)
    # Handle NaN (constant features)
    corr = np.nan_to_num(corr, nan=0.0)

    # Distance = 1 - |correlation|
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    # Force symmetry and non-negative (floating point)
    dist = (dist + dist.T) / 2.0
    dist = np.clip(dist, 0.0, 2.0)

    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")

    # Group features by cluster label
    clusters: dict[int, list[int]] = {}
    for feat_idx, cluster_id in zip(feature_indices, labels, strict=True):
        clusters.setdefault(int(cluster_id), []).append(feat_idx)

    return list(clusters.values())


# ── Main Pipeline ────────────────────────────────────────────────────────────


def neuronpedia_url(feature: int, layer: int = 22) -> str:
    """Generate Neuronpedia URL for a feature."""
    return f"https://www.neuronpedia.org/gemma-3-4b-it/{layer}-gemmascope-2-res-262k/{feature}"


def run_discrimination(
    sweep_dir: Path,
    output_path: Path,
    contrasts: list[str] | None = None,
    top_n: int = 500,
    cluster_threshold: float = 0.5,
    mode: str = "mean",
) -> dict[str, Any]:
    """Run full discrimination pipeline.

    Args:
        sweep_dir: Directory with judged JSONL files.
        output_path: Where to write results JSON.
        contrasts: Which contrasts to compute (default: both).
        top_n: Number of top features per contrast.
        cluster_threshold: Distance threshold for clustering.
        mode: Feature aggregation mode ("mean" or "max").

    Returns:
        Results dict (also written to output_path).
    """
    if contrasts is None:
        contrasts = ["A_vs_B", "A_vs_C"]

    print("Loading trials...")
    trials = load_trials(sweep_dir)
    print(f"Loaded {len(trials)} trials")

    # Filter to only judged trials
    judged = [t for t in trials if t.get("judge")]
    print(f"Judged: {len(judged)}")

    groups = assign_groups(judged)
    print(f"Group A (detected):   {len(groups['A'])}")
    print(f"Group B (control ok): {len(groups['B'])}")
    print(f"Group C (missed):     {len(groups['C'])}")
    print(f"Excluded:             {len(groups['excluded'])}")
    print()

    results: dict[str, Any] = {
        "sweep_dir": str(sweep_dir),
        "mode": mode,
        "top_n": top_n,
        "group_sizes": {k: len(v) for k, v in groups.items()},
        "contrasts": {},
    }

    for contrast in contrasts:
        if contrast == "A_vs_B":
            group_a, group_b, label = groups["A"], groups["B"], "detected_vs_control"
        elif contrast == "A_vs_C":
            group_a, group_b, label = groups["A"], groups["C"], "detected_vs_missed"
        else:
            print(f"Unknown contrast: {contrast}, skipping")
            continue

        if len(group_a) < 5 or len(group_b) < 5:
            print(f"Skipping {contrast}: too few trials (A={len(group_a)}, B={len(group_b)})")
            continue

        print(f"Computing {contrast} ({label})...")
        features = compute_contrast(group_a, group_b, mode=mode, top_n=top_n)
        print(f"  Top feature: #{features[0]['feature']} (d={features[0]['effect_size']:.2f})")

        # Cluster top features
        feature_indices = [f["feature"] for f in features]
        all_trials = group_a + group_b
        clusters_raw = cluster_features(
            all_trials, feature_indices, mode=mode, threshold=cluster_threshold,
        )

        # Build feature lookup for enrichment
        feat_lookup = {f["feature"]: f for f in features}

        clusters = []
        for i, members in enumerate(sorted(clusters_raw, key=lambda c: -len(c))):
            # Pick representative as highest abs effect size in cluster
            rep = max(members, key=lambda f: feat_lookup[f]["abs_effect_size"])
            cluster_entry = {
                "id": i,
                "representative": {
                    **feat_lookup[rep],
                    "layer": 22,
                    "neuronpedia_url": neuronpedia_url(rep),
                },
                "members": [
                    {**feat_lookup[m], "neuronpedia_url": neuronpedia_url(m)}
                    for m in sorted(members, key=lambda f: -feat_lookup[f]["abs_effect_size"])
                ],
            }
            clusters.append(cluster_entry)

        print(f"  {len(clusters)} clusters from {len(feature_indices)} features")

        results["contrasts"][contrast] = {
            "label": label,
            "n_group_a": len(group_a),
            "n_group_b": len(group_b),
            "top_features": features[:20],
            "clusters": clusters,
        }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="SAE feature discrimination")
    parser.add_argument("sweep_dir", type=Path, help="Directory with judged sweep JSONL")
    parser.add_argument("--output", type=Path, default=Path("results/discrimination.json"))
    parser.add_argument("--contrast", nargs="+", default=None,
                        choices=["A_vs_B", "A_vs_C"],
                        help="Contrasts to compute (default: both)")
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--cluster-threshold", type=float, default=0.5)
    parser.add_argument("--mode", choices=["mean", "max"], default="mean")
    args = parser.parse_args()

    if not args.sweep_dir.is_dir():
        print(f"Error: {args.sweep_dir} is not a directory")
        sys.exit(1)

    run_discrimination(
        sweep_dir=args.sweep_dir,
        output_path=args.output,
        contrasts=args.contrast,
        top_n=args.top_n,
        cluster_threshold=args.cluster_threshold,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
