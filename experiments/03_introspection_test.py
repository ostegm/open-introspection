#!/usr/bin/env python
"""Phase 3: Basic introspection testing.

Test if the model can detect injected concepts.

Usage:
    uv run python experiments/03_introspection_test.py              # default 3B
    uv run python experiments/03_introspection_test.py --model 7b
    uv run python experiments/03_introspection_test.py --model 14b  # uses 4-bit
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from open_introspection.introspection import (
    ExperimentResults,
    run_introspection_experiment,
)
from open_introspection.model import load_model

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

# Model size configurations
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "3b": {"name": "Qwen/Qwen2.5-3B-Instruct"},
    "7b": {"name": "Qwen/Qwen2.5-7B-Instruct"},
    # 14b requires ~28GB in bfloat16, and TransformerLens doesn't support quantization
}

# Target effective magnitude for injection (auto-scales strength per concept)
# 80 works well for 3B; same target should work across model sizes
TARGET_EFFECTIVE_MAGNITUDE = 80.0


def main() -> None:
    """Run basic introspection tests."""
    parser = argparse.ArgumentParser(description="Run introspection experiment")
    parser.add_argument(
        "--model",
        choices=["3b", "7b"],
        default="3b",
        help="Model size (default: 3b). 14b not supported (needs quantization).",
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

    # Use ~83% through model (layer 30 for 36-layer 3B was optimal in sweep)
    # Let run_introspection_experiment compute this from model.cfg.n_layers
    layer = None  # Auto-select based on model size

    print(f"\nRunning introspection experiment with {len(concepts)} concepts...")
    print(f"Target effective magnitude: {TARGET_EFFECTIVE_MAGNITUDE}")
    results: ExperimentResults = run_introspection_experiment(
        model,
        concepts=concepts,
        layer=layer,
        target_magnitude=TARGET_EFFECTIVE_MAGNITUDE,
    )

    # Analyze results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print("\n--- Control Trials (no injection) ---")
    for trial in results.control:
        response_preview: str = trial.response[-300:]  # Last part of response
        detected: bool = "yes" in response_preview.lower()[:100]
        status: str = (
            "DETECTED (false positive)" if detected else "Not detected (correct)"
        )
        print(f"\n{trial.concept}: {status}")
        print(f"  Response preview: {response_preview[:150]}...")

    print("\n--- Injection Trials ---")
    for trial in results.injection:
        response_preview = trial.response[-300:]
        detected = "yes" in response_preview.lower()[:100]
        concept_found: bool = trial.concept.lower() in response_preview.lower()
        detected_str: str = "DETECTED" if detected else "Not detected"
        identified_str: str = "IDENTIFIED" if concept_found else "Not identified"
        print(f"\n{trial.concept}: {detected_str}, {identified_str}")
        print(f"  Response preview: {response_preview[:150]}...")

    # Save results with config and timestamp
    output_dir: Path = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file: Path = output_dir / f"introspection_{timestamp}.json"

    # Compute actual layer used (same logic as run_introspection_experiment)
    actual_layer = layer if layer is not None else int(model.cfg.n_layers * 5 / 6)

    output_data: dict[str, Any] = {
        "config": {
            "model": model_name,
            "n_layers": model.cfg.n_layers,
            "layer": actual_layer,
            "target_magnitude": TARGET_EFFECTIVE_MAGNITUDE,
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
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
