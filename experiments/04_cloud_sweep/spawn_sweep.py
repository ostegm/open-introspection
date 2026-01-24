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
from datetime import datetime

import modal

# Model layer counts
MODEL_LAYERS = {
    "3b": 36,
    "7b": 28,
    "14b": 48,
    "32b": 64,
}

# GPU mapping
GPU_FUNCTIONS = {
    "3b": "run_sweep_l4",
    "7b": "run_sweep_l4",
    "14b": "run_sweep_a100",
    "32b": "run_sweep_a100_80gb",
}

DEFAULT_STRENGTHS = [1.5, 2.0, 2.5, 3.0, 4.0]
DEFAULT_TRIALS = 20
CONCEPTS = ["celebration", "ocean", "fear", "silence"]


def get_default_layers(model_size: str) -> list[int]:
    """Get default layers targeting 60-75% of model depth."""
    n_layers = MODEL_LAYERS[model_size]
    fractions = [0.60, 0.65, 0.70, 0.75]
    return sorted({int(n_layers * f) for f in fractions})


def main() -> None:
    parser = argparse.ArgumentParser(description="Spawn sweep jobs on deployed Modal app")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--model", choices=["3b", "7b", "14b", "32b"], default="3b")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS)
    parser.add_argument("--inject-style", choices=["all", "generation"], default="generation")
    parser.add_argument("--skip-judge", action="store_true", default=True)
    parser.add_argument("--with-judge", dest="skip_judge", action="store_false")
    parser.add_argument("--concepts", nargs="+", default=CONCEPTS,
                        help="Concepts to run (default: all 4)")
    args = parser.parse_args()

    experiment_id = args.experiment_id or datetime.now().strftime("sweep-%Y%m%d-%H%M%S")
    layers = args.layers or get_default_layers(args.model)

    # Calculate totals
    n_configs = len(layers) * len(args.strengths) * 2  # x2 for inject/control
    trials_per_concept = n_configs * args.trials
    total_trials = trials_per_concept * len(args.concepts)

    print("=" * 60)
    print(f"Spawning Sweep: {experiment_id}")
    print("=" * 60)
    print(f"Model: {args.model} ({MODEL_LAYERS[args.model]} layers)")
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
        return

    print(f"Spawning {len(args.concepts)} jobs on Modal...")
    print()

    # Spawn all jobs
    calls = []
    for concept in args.concepts:
        call = sweep_fn.spawn(
            concept=concept,
            model_size=args.model,
            trials=args.trials,
            experiment_id=experiment_id,
            inject_style=args.inject_style,
            skip_judge=args.skip_judge,
            layers=layers,
            strengths=args.strengths,
        )
        calls.append((concept, call))
        print(f"  {concept}: {call.object_id}")

    print()
    print("=" * 60)
    print("Jobs spawned successfully!")
    print("=" * 60)
    print()
    print("You can close this terminal - jobs run independently on Modal.")
    print()
    print("Monitor: https://modal.com/apps/ostegm/main/deployed/open-introspection-sweep")
    print(f"Results: gs://open-introspection-sweeps/{experiment_id}/")
    print()
    print("To download when complete:")
    print(f"  gsutil -m cp -r gs://open-introspection-sweeps/{experiment_id}/ data/sweeps/")


if __name__ == "__main__":
    main()
