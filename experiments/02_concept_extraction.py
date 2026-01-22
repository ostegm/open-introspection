#!/usr/bin/env python
"""Phase 2: Concept vector extraction.

Extract activation vectors for specific concepts and validate
by steering generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from open_introspection.concept_extraction import (
    DEFAULT_BASELINE_WORDS,
    extract_concept_vector,
    validate_concept_vector,
)
from open_introspection.model import load_model

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer


def main() -> None:
    """Extract and validate concept vectors."""
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

    for concept in concepts:
        print(f"\n--- Extracting vector for '{concept}' ---")

        vector: Tensor = extract_concept_vector(
            model,
            target_word=concept,
            baseline_words=DEFAULT_BASELINE_WORDS,
            layer=layer,
        )  # shape: (d_model,)

        print(f"Vector shape: {vector.shape}")
        print(f"Vector norm: {vector.norm().item():.4f}")

        # Validate by steering
        print(f"\nSteering with '{concept}' vector:")
        steered_output: str = validate_concept_vector(
            model,
            vector,
            concept,
            layer,
            injection_strength=2.0,
        )
        # Show just the generated part (after prompt)
        print(steered_output.split("weather.")[-1][:200])

    print("\n" + "=" * 60)
    print("Concept extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
