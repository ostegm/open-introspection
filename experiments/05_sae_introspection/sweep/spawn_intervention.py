#!/usr/bin/env python
"""Spawn SAE feature intervention experiment jobs on Modal.

Builds ablation/activation conditions from candidate features identified
in discrimination analysis. Supports phased execution (quick screen first,
then group-level drill-down if signal found).

Usage:
    # First deploy:
    uv run modal deploy experiments/05_sae_introspection/sweep/modal_app.py

    # Phase 1: Quick screen (320 trials)
    uv run python experiments/05_sae_introspection/sweep/spawn_intervention.py --phase 1

    # Phase 2: Individual groups (960 trials, run only if Phase 1 shows signal)
    uv run python experiments/05_sae_introspection/sweep/spawn_intervention.py --phase 2

    # Cherry-pick specific conditions:
    uv run python experiments/05_sae_introspection/sweep/spawn_intervention.py \
        --conditions ablate_all --trials 2 --concepts silence
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import modal
from google.cloud import storage

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ALL_CANDIDATE_FEATURES,
    BEST_INJECTION_LAYER,
    CANDIDATE_FEATURES,
    CODE_HASH,
    CONCEPTS,
    FEATURE_MEAN_A,
    GCS_BUCKET,
    FeatureIntervention,
    InterventionRequest,
    InterventionSpec,
)


def _build_ablation_spec(
    name: str, features: list[int], strength: float = 2.0,
) -> InterventionSpec:
    """Build an ablation condition: inject + zero target features."""
    return InterventionSpec(
        name=name,
        interventions=[
            FeatureIntervention(feature=f, mode="zero") for f in features
        ],
        inject=True,
        strength=strength,
    )


def _build_activation_spec(
    name: str, features: list[int],
) -> InterventionSpec:
    """Build an activation condition: no injection + clamp features to mean_A."""
    return InterventionSpec(
        name=name,
        interventions=[
            FeatureIntervention(feature=f, mode="set", value=FEATURE_MEAN_A[f])
            for f in features
        ],
        inject=False,
    )


def build_phase1_conditions() -> list[InterventionSpec]:
    """Phase 1: ablate_all + activate_all."""
    return [
        _build_ablation_spec("ablate_all", ALL_CANDIDATE_FEATURES),
        _build_activation_spec("activate_all", ALL_CANDIDATE_FEATURES),
    ]


def build_phase2_conditions() -> list[InterventionSpec]:
    """Phase 2: per-group ablations + activations + top1."""
    conditions = []
    for group_name, features in CANDIDATE_FEATURES.items():
        conditions.append(_build_ablation_spec(f"ablate_{group_name}", features))
    # Top-1 feature ablation
    conditions.append(_build_ablation_spec("ablate_top1", [14542]))
    # Per-group activations (perception + affect only, hedging is too many)
    for group_name in ["perception", "affect"]:
        features = CANDIDATE_FEATURES[group_name]
        conditions.append(_build_activation_spec(f"activate_{group_name}", features))
    return conditions


def gcs_path_exists(gcs_path: str) -> bool:
    """Check if a GCS path already exists."""
    try:
        parts = gcs_path.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return bool(blob.exists())
    except Exception as e:
        print(f"Warning: Could not check GCS path: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Spawn SAE intervention experiment")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=1,
        help="Phase 1 = quick screen, Phase 2 = per-group drill-down",
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Trials per condition per layer (default: 10)",
    )
    parser.add_argument(
        "--concepts", nargs="+", default=CONCEPTS,
        help=f"Concepts to run (default: {CONCEPTS})",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="Cherry-pick conditions by name (default: all for phase)",
    )
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing GCS outputs",
    )
    args = parser.parse_args()

    # Build conditions
    all_conditions = (
        build_phase1_conditions() if args.phase == 1
        else build_phase2_conditions()
    )

    # Filter if cherry-picking
    if args.conditions:
        selected = set(args.conditions)
        all_conditions = [c for c in all_conditions if c.name in selected]
        missing = selected - {c.name for c in all_conditions}
        if missing:
            all_names = [
                c.name for c in build_phase1_conditions() + build_phase2_conditions()
            ]
            print(f"ERROR: Unknown conditions: {missing}")
            print(f"Available: {all_names}")
            return 1

    if not all_conditions:
        print("ERROR: No conditions selected")
        return 1

    experiment_id = args.experiment_id or datetime.now().strftime(
        f"sae-intervention-p{args.phase}-%Y%m%d-%H%M%S"
    )

    # Build requests (one per concept)
    requests: list[InterventionRequest] = []
    for concept in args.concepts:
        gcs_path = f"gs://{GCS_BUCKET}/{experiment_id}/gemma-4b/{concept}.jsonl"
        requests.append(InterventionRequest(
            concept=concept,
            trials_per_condition=args.trials,
            conditions=all_conditions,
            experiment_id=experiment_id,
            gcs_path=gcs_path,
        ))

    # Check for existing outputs
    if not args.force:
        existing = [r.gcs_path for r in requests if gcs_path_exists(r.gcs_path)]
        if existing:
            print("ERROR: Output already exists (use --force to overwrite):")
            for path in existing:
                print(f"  {path}")
            return 1

    # Compute totals
    trials_per_concept = args.trials * len(all_conditions)
    total_trials = trials_per_concept * len(args.concepts)

    print("=" * 60)
    print(f"SAE Feature Intervention Experiment: {experiment_id}")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Concepts: {args.concepts}")
    print(f"Trials per condition: {args.trials}")
    print(f"Injection layer: {BEST_INJECTION_LAYER}")
    print()
    print("Conditions:")
    for c in all_conditions:
        n = args.trials * len(args.concepts)
        feat_str = ", ".join(f"#{iv.feature}" for iv in c.interventions)
        print(f"  {c.name}: {len(c.interventions)} features [{feat_str}] "
              f"inject={c.inject} -> {n} trials")
    print()
    print(f"Total trials: {total_trials}")
    print()

    # Look up deployed function
    try:
        intervention_fn = modal.Function.from_name(
            "sae-feature-discovery-sweep", "run_intervention_l4",
        )
    except modal.exception.NotFoundError:
        print("ERROR: App not deployed (or missing run_intervention_l4). Run:")
        print("  uv run modal deploy experiments/05_sae_introspection/sweep/modal_app.py")
        return 1

    # Version check
    try:
        hash_fn = modal.Function.from_name("sae-feature-discovery-sweep", "get_code_hash")
        deployed_hash = hash_fn.remote()
        if deployed_hash != CODE_HASH:
            print(f"ERROR: Deployed app is stale (hash {deployed_hash} vs local {CODE_HASH})")
            print("Run: uv run modal deploy experiments/05_sae_introspection/sweep/modal_app.py")
            return 1
    except modal.exception.NotFoundError:
        print("WARNING: Deployed app missing version check - redeploy recommended")

    print(f"Spawning {len(requests)} jobs on Modal...")
    print()

    calls = []
    for request in requests:
        call = intervention_fn.spawn(request=request.model_dump())
        calls.append((request.concept, call))
        print(f"  {request.concept}: {call.object_id}")

    print()
    print("=" * 60)
    print("Jobs spawned successfully!")
    print("=" * 60)
    print()
    print("You can close this terminal - jobs run independently on Modal.")
    print()
    print("Monitor: https://modal.com/apps/ostegm/main/deployed/sae-feature-discovery-sweep")
    print(f"Results: gs://{GCS_BUCKET}/{experiment_id}/")
    print()
    print("To download when complete:")
    print(f"  gsutil -m cp -r gs://{GCS_BUCKET}/{experiment_id}/ data/sweeps/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
