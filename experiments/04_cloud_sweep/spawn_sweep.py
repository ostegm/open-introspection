#!/usr/bin/env python
"""Spawn sweep jobs on the deployed Modal app.

This script spawns jobs that run independently on Modal - you can close
your terminal and the jobs will continue running.

Usage:
    # First deploy the app (only needed once or after code changes):
    uv run modal deploy experiments/04_cloud_sweep/modal_app.py

    # Then spawn jobs:
    uv run python experiments/04_cloud_sweep/spawn_sweep.py --trials 20
    uv run python experiments/04_cloud_sweep/spawn_sweep.py --model 7b --trials 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import modal
from google.cloud import storage

# Add paths for imports (must happen before local imports)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    CONCEPTS,
    DEFAULT_STRENGTHS,
    DEFAULT_TRIALS,
    GCS_BUCKET,
    GPU_FUNCTIONS,
    MODEL_CONFIGS,
    SweepRequest,
    get_default_layers,
)
from src.open_introspection.introspection import PROMPT_MESSAGES


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
    parser = argparse.ArgumentParser(description="Spawn sweep jobs on deployed Modal app")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--model", choices=["3b", "7b", "14b", "32b"], default="3b")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS)
    parser.add_argument("--inject-style", choices=["all", "generation"], default="generation")
    parser.add_argument("--prompt-version", default="v2", help="Prompt version (default: v2)")
    parser.add_argument("--concepts", nargs="+", default=CONCEPTS,
                        help="Concepts to run (default: all 4)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing GCS outputs")
    args = parser.parse_args()

    experiment_id = args.experiment_id or datetime.now().strftime("sweep-%Y%m%d-%H%M%S")
    layers = args.layers or get_default_layers(args.model)
    n_layers = int(MODEL_CONFIGS[args.model]["n_layers"])

    # Validate prompt version (fail fast)
    if args.prompt_version not in PROMPT_MESSAGES:
        valid = ", ".join(sorted(PROMPT_MESSAGES.keys()))
        print(f"ERROR: Invalid prompt version '{args.prompt_version}'")
        print(f"Valid options: {valid}")
        return 1

    # Build requests for each concept
    requests: list[SweepRequest] = []
    for concept in args.concepts:
        gcs_path = f"gs://{GCS_BUCKET}/{experiment_id}/{args.model}/{concept}.jsonl"
        requests.append(SweepRequest(
            concept=concept,
            model_size=args.model,
            trials=args.trials,
            experiment_id=experiment_id,
            inject_style=args.inject_style,
            layers=layers,
            strengths=args.strengths,
            gcs_path=gcs_path,
            prompt_version=args.prompt_version,
        ))

    # Check for existing outputs (fail fast)
    if not args.force:
        existing = []
        for req in requests:
            if gcs_path_exists(req.gcs_path):
                existing.append(req.gcs_path)
        if existing:
            print("ERROR: Output already exists (use --force to overwrite):")
            for path in existing:
                print(f"  {path}")
            return 1

    # Calculate totals
    n_configs = len(layers) * len(args.strengths) * 2  # x2 for inject/control
    trials_per_concept = n_configs * args.trials
    total_trials = trials_per_concept * len(args.concepts)

    print("=" * 60)
    print(f"Spawning Sweep: {experiment_id}")
    print("=" * 60)
    print(f"Model: {args.model} ({n_layers} layers)")
    print(f"GPU: {GPU_FUNCTIONS[args.model]}")
    print(f"Layers: {layers}")
    print(f"Strengths: {args.strengths}")
    print(f"Trials per config: {args.trials}")
    print(f"Concepts: {args.concepts}")
    print()
    print(f"Trials per concept: {trials_per_concept}")
    print(f"Total trials: {total_trials}")
    print()

    # Look up the deployed function
    fn_name = GPU_FUNCTIONS[args.model]
    try:
        sweep_fn = modal.Function.from_name("open-introspection-sweep", fn_name)
    except modal.exception.NotFoundError:
        print("ERROR: App not deployed. Run this first:")
        print("  uv run modal deploy experiments/04_cloud_sweep/modal_app.py")
        return 1

    print(f"Spawning {len(requests)} jobs on Modal...")
    print()

    # Spawn all jobs
    calls = []
    for request in requests:
        # Pass as dict for Modal serialization
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
    print("Monitor: https://modal.com/apps/ostegm/main/deployed/open-introspection-sweep")
    print(f"Results: gs://{GCS_BUCKET}/{experiment_id}/")
    print()
    print("To download when complete:")
    print(f"  gsutil -m cp -r gs://{GCS_BUCKET}/{experiment_id}/ data/sweeps/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
