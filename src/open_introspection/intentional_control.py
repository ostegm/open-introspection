"""Phase 6: Intentional control - can models control their internal representations?"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel

from open_introspection.concept_extraction import (
    DEFAULT_BASELINE_WORDS,
    extract_concept_vector,
)

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer


class ControlExperimentResult(BaseModel):
    """Result from an intentional control experiment."""

    concept: str
    layer: int
    think: float
    dont_think: float
    control: float
    think_vs_control: float
    dont_think_vs_control: float
    think_vs_dont_think: float
    shows_control: bool


def measure_concept_activation(
    model: HookedTransformer,
    prompt: str,
    concept_vector: Tensor,
    layer: int,
) -> float:
    """
    Measure how much a concept is represented in activations.

    Args:
        model: HookedTransformer model
        prompt: Text prompt to measure activations for
        concept_vector: The concept vector to compare against (shape: d_model)
        layer: Layer to measure at

    Returns:
        Cosine similarity between response activations and concept vector
    """
    tokens = model.to_tokens(prompt)  # shape: (batch, seq)
    _, cache = model.run_with_cache(tokens)

    activations: Tensor = cache[f"blocks.{layer}.hook_resid_post"][
        0
    ]  # shape: (seq, d_model)

    # Compute cosine similarity across positions
    similarities: Tensor = torch.nn.functional.cosine_similarity(
        activations,
        concept_vector.unsqueeze(0),
        dim=-1,
    )  # shape: (seq,)

    return similarities.mean().item()


def run_intentional_control_experiment(
    model: HookedTransformer,
    concept: str,
    base_sentence: str = "I am writing a simple sentence about the park.",
    layer: int | None = None,
    baseline_words: list[str] | None = None,
) -> ControlExperimentResult:
    """
    Test if model can control its internal concept representations.

    Compares activations when instructed to:
    1. Think about the concept
    2. NOT think about the concept
    3. Control (no mention of concept)

    Args:
        model: HookedTransformer model
        concept: Concept to test control over
        base_sentence: Sentence to write in all conditions
        layer: Layer to measure at
        baseline_words: Words for baseline

    Returns:
        Dict with similarity scores for each condition
    """
    if layer is None:
        layer = int(model.cfg.n_layers * 2 / 3)

    if baseline_words is None:
        baseline_words = DEFAULT_BASELINE_WORDS

    concept_vector: Tensor = extract_concept_vector(
        model, concept, baseline_words, layer
    )  # shape: (d_model,)

    # Construct prompts
    think_prompt: str = (
        f"Write the following sentence while thinking about '{concept}': {base_sentence}"
    )
    dont_think_prompt: str = (
        f"Write the following sentence WITHOUT thinking about '{concept}': {base_sentence}"
    )
    control_prompt: str = f"Write the following sentence: {base_sentence}"

    # Measure activations
    think_score: float = measure_concept_activation(
        model, think_prompt, concept_vector, layer
    )
    dont_think_score: float = measure_concept_activation(
        model, dont_think_prompt, concept_vector, layer
    )
    control_score: float = measure_concept_activation(
        model, control_prompt, concept_vector, layer
    )

    # Analysis
    think_vs_control: float = think_score - control_score
    dont_think_vs_control: float = dont_think_score - control_score
    think_vs_dont_think: float = think_score - dont_think_score

    # Expected: think > dont_think > control (ideally)
    shows_control: bool = think_score > dont_think_score

    return ControlExperimentResult(
        concept=concept,
        layer=layer,
        think=think_score,
        dont_think=dont_think_score,
        control=control_score,
        think_vs_control=think_vs_control,
        dont_think_vs_control=dont_think_vs_control,
        think_vs_dont_think=think_vs_dont_think,
        shows_control=shows_control,
    )


def run_control_experiment_batch(
    model: HookedTransformer,
    concepts: list[str],
    layer: int | None = None,
) -> list[ControlExperimentResult]:
    """
    Run intentional control experiment for multiple concepts.

    Args:
        model: HookedTransformer model
        concepts: List of concepts to test
        layer: Layer to measure at

    Returns:
        List of result dicts for each concept
    """
    results: list[ControlExperimentResult] = []
    for concept in concepts:
        print(f"Testing control for concept: {concept}")
        result: ControlExperimentResult = run_intentional_control_experiment(
            model, concept, layer=layer
        )
        results.append(result)
    return results
