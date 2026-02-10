#!/usr/bin/env python
"""Phase 2: Concept vector extraction.

Extract activation vectors for specific concepts and validate
by steering generation.
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


def main() -> None:
    """Extract and validate concept vectors."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("Phase 2: Concept Vector Extraction")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model: HookedTransformer = load_model(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dtype=torch.bfloat16,
    )

    layer: int = int(model.cfg.n_layers * 2 / 3)
    print(f"Using layer {layer} (2/3 through {model.cfg.n_layers} layers)")

    # Test concepts
    concepts: list[str] = ["ocean", "music", "fear", "celebration", "silence"]

    # Pre-compute baseline mean once (excludes all test concepts)
    print("\nComputing baseline mean (50 words, excluding test concepts)...")
    baseline_mean: Tensor = compute_baseline_mean(
        model,
        layer=layer,
        exclude_words=concepts,
    )
    print(f"Baseline mean shape: {baseline_mean.shape}")

    # Extract all concept vectors first
    print("\n" + "=" * 60)
    print("Extracting concept vectors")
    print("=" * 60)
    vectors: dict[str, Tensor] = {}
    for concept in concepts:
        vector: Tensor = extract_concept_vector(
            model,
            target_word=concept,
            layer=layer,
            cached_baseline_mean=baseline_mean,
        )
        vectors[concept] = vector
        print(f"  {concept}: norm={vector.norm().item():.2f}")

    # Compute cosine similarity matrix
    print("\n" + "=" * 60)
    print("Cosine similarity between concept vectors")
    print("=" * 60)
    print(f"{'':12}", end="")
    for c in concepts:
        print(f"{c:>10}", end="")
    print()
    for c1 in concepts:
        print(f"{c1:12}", end="")
        for c2 in concepts:
            v1, v2 = vectors[c1], vectors[c2]
            cos_sim = torch.nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)
            ).item()
            print(f"{cos_sim:>10.3f}", end="")
        print()

    # Validate by steering (temperature=1.0 per paper methodology for trials)
    print("\n" + "=" * 60)
    print("Steering validation (temperature=1.0)")
    print("=" * 60)
    iterations = 5
    for concept in concepts:
        print(f"\n--- {concept} (n={iterations}) ---")
        for n in range(iterations):
            steered_output: str = validate_concept_vector(
                model,
                vectors[concept],
                concept,
                layer,
                injection_strength=2.0,
                temperature=1.0,
            )
            # Show just the generated part
            print(f"  t{n}: {steered_output[:150]}")

    print("\n" + "=" * 60)
    print("Concept extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
