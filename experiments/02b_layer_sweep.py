#!/usr/bin/env python
"""Layer sweep experiment.

Test concept extraction and steering across different layers to find
optimal injection points for different concepts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from open_introspection.concept_extraction import (
    compute_baseline_mean,
    extract_concept_vector,
    validate_concept_vector,
)
from open_introspection.model import load_model

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer


def run_layer_sweep(
    model: HookedTransformer,
    concepts: list[str],
    layers: list[int],
    trials_per_layer: int = 5,
    injection_strength: float = 2.0,
) -> dict[str, dict[int, list[str]]]:
    """
    Run steering trials across multiple layers for each concept.

    Returns:
        Nested dict: results[concept][layer] = list of generated outputs
    """
    results: dict[str, dict[int, list[str]]] = {c: {} for c in concepts}

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print("=" * 60)

        # Compute baseline mean for this layer
        baseline_mean: Tensor = compute_baseline_mean(
            model,
            layer=layer,
            exclude_words=concepts,
        )

        for concept in concepts:
            # Extract concept vector at this layer
            vector: Tensor = extract_concept_vector(
                model,
                target_word=concept,
                layer=layer,
                cached_baseline_mean=baseline_mean,
            )

            print(f"\n{concept} (norm={vector.norm().item():.2f}):")

            # Run steering trials
            outputs: list[str] = []
            for t in range(trials_per_layer):
                output: str = validate_concept_vector(
                    model,
                    vector,
                    concept,
                    layer,
                    injection_strength=injection_strength,
                    temperature=1.0,
                )
                # Truncate for display
                short = output[:80].replace("\n", " ")
                print(f"  t{t}: {short}")
                outputs.append(output)

            results[concept][layer] = outputs

    return results


def main() -> None:
    """Run layer sweep experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("Layer Sweep Experiment")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model: HookedTransformer = load_model(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype=torch.bfloat16,
    )

    n_layers = model.cfg.n_layers
    print(f"Model has {n_layers} layers")

    # Test layers: early, mid-early, mid, mid-late, late
    # For 36 layers: [6, 12, 18, 24, 30, 34]
    layers = [
        n_layers // 6,      # ~17% (early)
        n_layers // 3,      # ~33% (mid-early)
        n_layers // 2,      # 50% (mid)
        2 * n_layers // 3,  # ~67% (mid-late, current default)
        5 * n_layers // 6,  # ~83% (late)
        n_layers - 2,       # near-final
    ]
    print(f"Testing layers: {layers}")

    # Concepts to test:
    # - ocean: broken at layer 24, want to find where it works
    # - fear: working at layer 24, control to verify method
    concepts = ["ocean", "fear"]

    run_layer_sweep(
        model,
        concepts=concepts,
        layers=layers,
        trials_per_layer=5,
        injection_strength=2.0,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nManually review outputs above for hit rates.")
    print("Look for:")
    print("  - ocean: water, sea, waves, marine, beach, etc.")
    print("  - fear: afraid, scared, fear, anxiety, dread, etc.")


if __name__ == "__main__":
    main()
