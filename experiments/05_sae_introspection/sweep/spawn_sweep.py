#!/usr/bin/env python
"""Spawn SAE feature discovery sweep jobs on Modal.

Spawns 4 parallel jobs (one per concept) that run independently.

Usage:
    # First deploy:
    uv run modal deploy experiments/05_sae_introspection/sweep/modal_app.py

    # Then spawn:
    uv run python experiments/05_sae_introspection/sweep/spawn_sweep.py
    uv run python experiments/05_sae_introspection/sweep/spawn_sweep.py --trials 10
    uv run python experiments/05_sae_introspection/sweep/spawn_sweep.py --concepts ocean fear
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
    CODE_HASH,
    CONCEPTS,
    CONTROL_TRIALS_PER_CONCEPT,
    DEFAULT_TRIALS,
    GCS_BUCKET,
    INJECTION_LAYERS,
    STRENGTHS,
    SweepRequest,
)


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
    parser = argparse.ArgumentParser(description="Spawn SAE feature discovery sweep")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Trials per injection config (default: {DEFAULT_TRIALS})")
    parser.add_argument("--control-trials", type=int, default=CONTROL_TRIALS_PER_CONCEPT,
                        help=f"Control trials per concept (default: {CONTROL_TRIALS_PER_CONCEPT})")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--concepts", nargs="+", default=CONCEPTS,
                        help=f"Concepts to run (default: {CONCEPTS})")
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--inject-style", choices=["all", "generation"], default="generation")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing GCS outputs")
    args = parser.parse_args()

    experiment_id = args.experiment_id or datetime.now().strftime("sae-sweep-%Y%m%d-%H%M%S")

    # Build requests
    requests: list[SweepRequest] = []
    for concept in args.concepts:
        gcs_path = f"gs://{GCS_BUCKET}/{experiment_id}/gemma-4b/{concept}.jsonl"
        requests.append(SweepRequest(
            concept=concept,
            trials=args.trials,
            experiment_id=experiment_id,
            gcs_path=gcs_path,
            inject_style=args.inject_style,
            prompt_version=args.prompt_version,
            control_trials=args.control_trials,
        ))

    # Check for existing outputs
    if not args.force:
        existing = [r.gcs_path for r in requests if gcs_path_exists(r.gcs_path)]
        if existing:
            print("ERROR: Output already exists (use --force to overwrite):")
            for path in existing:
                print(f"  {path}")
            return 1

    # Calculate totals
    n_injection = len(INJECTION_LAYERS) * len(STRENGTHS) * args.trials
    n_per_concept = n_injection + args.control_trials
    total = n_per_concept * len(args.concepts)

    print("=" * 60)
    print(f"SAE Feature Discovery Sweep: {experiment_id}")
    print("=" * 60)
    print("Model: google/gemma-3-4b-it")
    print("SAE: gemma-scope-2-4b-it-res / layer_22_width_262k_l0_small")
    print(f"Injection layers: {INJECTION_LAYERS}")
    print(f"Strengths: {STRENGTHS}")
    print(f"Trials per injection config: {args.trials}")
    print(f"Control trials per concept: {args.control_trials}")
    print(f"Concepts: {args.concepts}")
    print()
    print(f"Injection trials per concept: {n_injection}")
    print(f"Trials per concept: {n_per_concept}")
    print(f"Total trials: {total}")
    print()

    # Look up deployed function
    try:
        sweep_fn = modal.Function.from_name("sae-feature-discovery-sweep", "run_sweep_l4")
    except modal.exception.NotFoundError:
        print("ERROR: App not deployed. Run:")
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
        call = sweep_fn.spawn(request=request.model_dump())
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
