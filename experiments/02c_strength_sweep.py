#!/usr/bin/env python
"""Injection strength sweep experiment.

Test how injection strength affects steering quality.
Key question: Should strength be normalized by vector norm?
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


def run_strength_sweep(
    model: HookedTransformer,
    concepts: list[str],
    strengths: list[float],
    layer: int,
    trials_per_strength: int = 3,
) -> None:
    """Run steering trials across multiple injection strengths."""
    # Compute baseline mean once
    baseline_mean: Tensor = compute_baseline_mean(
        model,
        layer=layer,
        exclude_words=concepts,
    )

    # Extract concept vectors
    vectors: dict[str, Tensor] = {}
    norms: dict[str, float] = {}
    for concept in concepts:
        vector: Tensor = extract_concept_vector(
            model,
            target_word=concept,
            layer=layer,
            cached_baseline_mean=baseline_mean,
        )
        vectors[concept] = vector
        norms[concept] = vector.norm().item()
        print(f"  {concept}: norm={norms[concept]:.2f}")

    # Test each strength
    for strength in strengths:
        print(f"\n{'='*60}")
        print(f"Injection Strength: {strength}")
        print("=" * 60)

        for concept in concepts:
            vector = vectors[concept]
            effective_magnitude = strength * norms[concept]
            print(f"\n{concept} (effective magnitude: {effective_magnitude:.1f}):")

            for t in range(trials_per_strength):
                output: str = validate_concept_vector(
                    model,
                    vector,
                    concept,
                    layer,
                    injection_strength=strength,
                    temperature=1.0,
                )
                short = output[:100].replace("\n", " ")
                print(f"  t{t}: {short}")


def main() -> None:
    """Run injection strength sweep."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("Injection Strength Sweep")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model: HookedTransformer = load_model(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype=torch.bfloat16,
    )

    # Use layer 18 (good balanced results from layer sweep)
    layer = 18
    print(f"Using layer {layer}")

    # Test concepts: one that works (fear), one that's marginal (ocean at this layer)
    concepts = ["fear", "ocean"]
    print(f"Testing concepts: {concepts}")
    print("\nExtracting vectors...")

    # Strengths to test: from subtle to overwhelming
    # Based on layer sweep: norm ~16-18 at layer 18
    # At strength 2.0, effective magnitude ~32-36
    # At layer 34, norm was ~85-95, causing degeneracy
    strengths = [0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"Testing strengths: {strengths}")

    run_strength_sweep(
        model,
        concepts=concepts,
        strengths=strengths,
        layer=layer,
        trials_per_strength=3,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nExpected pattern:")
    print("  - 0.5: Too weak, minimal effect")
    print("  - 1.0-2.0: Sweet spot, concept-related outputs")
    print("  - 5.0: Strong steering, may start seeing artifacts")
    print("  - 10.0: Possibly degenerate (like layer 34 fear)")
    print("\nCompare effective magnitude (strength * norm) across conditions.")


if __name__ == "__main__":
    main()
