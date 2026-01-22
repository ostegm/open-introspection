#!/usr/bin/env python
"""Phase 3: Basic introspection testing.

Test if the model can detect injected concepts.
"""

from __future__ import annotations

import json
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


def main() -> None:
    """Run basic introspection tests."""
    print("=" * 60)
    print("Phase 3: Basic Introspection Test")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model: HookedTransformer = load_model(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype=torch.float16,
    )

    # Run experiment
    concepts: list[str] = ["ocean", "music", "fear", "celebration", "silence"]

    print(f"\nRunning introspection experiment with {len(concepts)} concepts...")
    results: ExperimentResults = run_introspection_experiment(
        model,
        concepts=concepts,
        injection_strength=2.0,
    )

    # Analyze results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print("\n--- Control Trials (no injection) ---")
    for trial in results["control"]:
        response_preview: str = trial["response"][-300:]  # Last part of response
        detected: bool = "yes" in response_preview.lower()[:100]
        status: str = (
            "DETECTED (false positive)" if detected else "Not detected (correct)"
        )
        print(f"\n{trial['concept']}: {status}")
        print(f"  Response preview: {response_preview[:150]}...")

    print("\n--- Injection Trials ---")
    for trial in results["injection"]:
        response_preview = trial["response"][-300:]
        detected = "yes" in response_preview.lower()[:100]
        concept_found: bool = trial["concept"].lower() in response_preview.lower()
        detected_str: str = "DETECTED" if detected else "Not detected"
        identified_str: str = "IDENTIFIED" if concept_found else "Not identified"
        print(f"\n{trial['concept']}: {detected_str}, {identified_str}")
        print(f"  Response preview: {response_preview[:150]}...")

    # Save results
    output_dir: Path = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_file: Path = output_dir / "introspection_results.json"

    # Convert to serializable dict
    serializable_results: dict[str, Any] = {
        "control": list(results["control"]),
        "injection": list(results["injection"]),
    }

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
