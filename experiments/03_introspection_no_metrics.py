#!/usr/bin/env python
"""Phase 3: Basic introspection testing.

Test if the model can detect injected concepts.

Usage:
    uv run python experiments/03_introspection_test.py
    uv run python experiments/03_introspection_test.py --model 7b
    uv run python experiments/03_introspection_test.py --layer 30 --strength 2.5
    uv run python experiments/03_introspection_test.py --prompt v1 --magnitude 80
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from open_introspection.introspection import (
    TEMPERATURE,
    ExperimentResults,
    run_introspection_experiment,
)
from open_introspection.model import load_model

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

# Model size configurations
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "3b": {"name": "Qwen/Qwen2.5-3B-Instruct", "n_layers": 36},
    "7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "n_layers": 28},
    # 14b requires ~28GB in bfloat16, and TransformerLens doesn't support quantization
}

# Default target effective magnitude for injection (auto-scales strength per concept)
# 70-100 works well for 3B based on sweep experiments
DEFAULT_TARGET_MAGNITUDE = 70.0


def main() -> None:
    """Run basic introspection tests."""
    parser = argparse.ArgumentParser(description="Run introspection experiment")
    parser.add_argument(
        "--model",
        choices=["3b", "7b"],
        default="3b",
        help="Model size (default: 3b). 14b not supported (needs quantization).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to inject at (default: ~2/3 through model, e.g. 24 for 3B).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Raw injection strength multiplier (bypasses auto-scaling). "
        "Typical values: 2.0-3.0. If not set, uses --magnitude for auto-scaling.",
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=DEFAULT_TARGET_MAGNITUDE,
        help=f"Target effective magnitude for auto-scaling (default: {DEFAULT_TARGET_MAGNITUDE}). "
        "Ignored if --strength is set.",
    )
    parser.add_argument(
        "--prompt",
        choices=["v1", "v2"],
        default="v2",
        help="Prompt version to use (default: v2).",
    )
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    model_name: str = config["name"]

    print("=" * 60)
    print("Phase 3: Basic Introspection Test")
    print("=" * 60)
    print(f"Model: {model_name}")

    # Load model
    print("\nLoading model...")
    model: HookedTransformer = load_model(
        model_name=model_name,
        dtype=torch.bfloat16,
    )

    # Run experiment
    # Concepts ordered by reliability from exp 02:
    # - celebration: 3/3 hits (most reliable)
    # - ocean: 3/3 hits at layer 30
    # - fear: 2-3/3 hits (noisier)
    # - silence: 5/5 at layer 24, untested at 30
    # Note: "music" excluded - only 1/5 hits, produces cats instead
    concepts: list[str] = ["celebration", "ocean", "fear", "silence"]

    # Layer selection: use CLI arg, or auto-select based on model size
    layer = args.layer  # None = auto-select in run_introspection_experiment

    print(f"\nRunning introspection experiment with {len(concepts)} concepts...")
    print(f"Prompt version: {args.prompt}")
    if args.strength is not None:
        print(f"Fixed injection strength: {args.strength}")
    else:
        print(f"Target effective magnitude: {args.magnitude} (auto-scaling)")
    results: ExperimentResults = run_introspection_experiment(
        model,
        concepts=concepts,
        layer=layer,
        target_magnitude=args.magnitude,
        injection_strength=args.strength,
        prompt_version=args.prompt,
    )

    # Save results with config and timestamp
    output_dir: Path = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file: Path = output_dir / f"introspection_{timestamp}.json"

    # Compute actual layer used (same logic as run_introspection_experiment)
    actual_layer = layer if layer is not None else int(model.cfg.n_layers * 2 / 3)

    output_data: dict[str, Any] = {
        "config": {
            "model": model_name,
            "n_layers": model.cfg.n_layers,
            "layer": actual_layer,
            "target_magnitude": args.magnitude,
            "injection_strength": args.strength,
            "prompt_version": args.prompt,
            "temperature": TEMPERATURE,
            "concepts": concepts,
            "timestamp": timestamp,
        },
        "results": {
            "control": [t.model_dump() for t in results.control],
            "injection": [t.model_dump() for t in results.injection],
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print("\n" + "=" * 60)
    print(f"Results Summary written to {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
