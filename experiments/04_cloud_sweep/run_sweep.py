#!/usr/bin/env python
"""Run parallel introspection sweep on Modal.

Sweep design based on EXPERIMENT_LOG.md findings:
- Previous generation-style test: 0% introspection at layers 20,24 / strengths 2-3
- Hypothesis: need higher strengths or different layers for generation-only injection
- This sweep tests wider range to find if any config produces introspection

Usage:
    uv run python experiments/04_cloud_sweep/run_sweep.py
    uv run python experiments/04_cloud_sweep/run_sweep.py --trials 5  # quick test
"""

from __future__ import annotations

import argparse
from datetime import datetime

# Default sweep parameters
DEFAULT_LAYERS = [20, 22, 24, 26, 28]
DEFAULT_STRENGTHS = [1.5, 2.0, 2.5, 3.0, 4.0]
DEFAULT_TRIALS = 20
CONCEPTS = ["celebration", "ocean", "fear", "silence"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parallel introspection sweep")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--model", choices=["3b", "7b"], default="3b")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=DEFAULT_STRENGTHS,
    )
    parser.add_argument(
        "--inject-style",
        choices=["all", "generation"],
        default="generation",
    )
    parser.add_argument("--skip-judge", action="store_true", default=True)
    parser.add_argument("--with-judge", dest="skip_judge", action="store_false")
    args = parser.parse_args()

    experiment_id = args.experiment_id or datetime.now().strftime(
        "generation-sweep-%Y%m%d-%H%M%S"
    )

    n_layers = len(args.layers)
    n_strengths = len(args.strengths)
    trials_per_concept = n_layers * n_strengths * args.trials * 2
    total_trials = trials_per_concept * len(CONCEPTS)

    print("=" * 50)
    print(f"Introspection Sweep: {experiment_id}")
    print("=" * 50)
    print(f"Layers: {args.layers}")
    print(f"Strengths: {args.strengths}")
    print(f"Trials per config: {args.trials}")
    print(f"Inject style: {args.inject_style}")
    print(f"Judge: {'skipped' if args.skip_judge else 'enabled'}")
    print()
    print(f"Trials per concept: {trials_per_concept}")
    print(f"Total trials: {total_trials}")
    print()
    print(f"Launching {len(CONCEPTS)} parallel GPU workers...")
    print()

    # Launch via subprocess so each worker runs independently
    import subprocess
    import sys

    procs = []
    for concept in CONCEPTS:
        cmd = [
            sys.executable, "-m", "modal", "run",
            "experiments/04_cloud_sweep/modal_app.py",
            "--concept", concept,
            "--model", args.model,
            "--trials", str(args.trials),
            "--experiment-id", experiment_id,
            "--inject-style", args.inject_style,
            "--layers", ",".join(str(layer) for layer in args.layers),
            "--strengths", ",".join(str(s) for s in args.strengths),
        ]
        if args.skip_judge:
            cmd.append("--skip-judge")

        print(f"  Starting {concept}...")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((concept, proc))

    print()
    print("=" * 50)
    print("Workers launched (running in background)")
    print("=" * 50)
    print()
    print("You can close this terminal - workers run on Modal.")
    print("Or wait here to see results...")
    print()
    print("Monitor: https://modal.com/apps")
    print(f"GCS bucket: gs://open-introspection-sweeps/{experiment_id}/")
    print()

    # Wait for all and print results
    for concept, proc in procs:
        proc.wait()
        print(f"{concept}: exit code {proc.returncode}")

    print()
    print("To download:")
    print(f"  gsutil -m cp -r gs://open-introspection-sweeps/{experiment_id}/ data/sweeps/")


if __name__ == "__main__":
    main()
