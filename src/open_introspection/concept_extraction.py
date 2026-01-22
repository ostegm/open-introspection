"""Phase 2: Concept vector extraction using mean subtraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer
    from transformer_lens.hook_points import HookPoint

# Default baseline words for concept extraction
DEFAULT_BASELINE_WORDS: list[str] = [
    "apple",
    "river",
    "mountain",
    "music",
    "friendship",
    "computer",
    "sunset",
    "democracy",
    "coffee",
    "science",
]


def extract_concept_vector(
    model: HookedTransformer,
    target_word: str,
    baseline_words: list[str] | None = None,
    layer: int | None = None,
    token_position: int = -1,
) -> Tensor:
    """
    Extract a concept vector using mean subtraction.

    Following the paper's method:
    1. Get activations for "Tell me about {target_word}"
    2. Get activations for "Tell me about {baseline_word}" for each baseline
    3. Subtract mean of baselines from target

    Args:
        model: HookedTransformer model
        target_word: The concept to extract a vector for
        baseline_words: Words to use as baseline (default: DEFAULT_BASELINE_WORDS)
        layer: Layer to extract from (default: ~2/3 through model)
        token_position: Token position to extract from (-1 = last token)

    Returns:
        Concept vector as a tensor of shape (d_model,)
    """
    if baseline_words is None:
        baseline_words = DEFAULT_BASELINE_WORDS

    if layer is None:
        # Default to ~2/3 through the model
        layer = int(model.cfg.n_layers * 2 / 3)

    def get_activation(word: str) -> Tensor:
        prompt = f"Tell me about {word}."
        tokens = model.to_tokens(prompt)  # shape: (batch, seq)
        _, cache = model.run_with_cache(tokens)
        # Get residual stream at specified layer and position
        activation: Tensor = cache[f"blocks.{layer}.hook_resid_post"][
            0, token_position, :
        ]  # shape: (d_model,)
        return activation

    # Get target activation
    target_act: Tensor = get_activation(target_word)  # shape: (d_model,)

    # Get baseline activations
    baseline_acts: Tensor = torch.stack(
        [get_activation(w) for w in baseline_words]
    )  # shape: (n_baselines, d_model)
    baseline_mean: Tensor = baseline_acts.mean(dim=0)  # shape: (d_model,)

    # Concept vector = target - baseline mean
    concept_vector: Tensor = target_act - baseline_mean  # shape: (d_model,)

    return concept_vector


def validate_concept_vector(
    model: HookedTransformer,
    concept_vector: Tensor,
    concept_name: str,  # noqa: ARG001
    layer: int,
    injection_strength: float = 2.0,
) -> str:
    """
    Validate a concept vector by steering generation.

    Inject the vector and see if outputs become more related to the concept.

    Args:
        model: HookedTransformer model
        concept_vector: The concept vector to inject (shape: d_model)
        concept_name: Name of the concept (for logging)
        layer: Layer to inject at
        injection_strength: How strongly to inject (multiplier)

    Returns:
        Generated text with the concept vector injected
    """
    prompt = "Write a short sentence about the weather."

    def injection_hook(
        activation: Tensor,
        hook: HookPoint,  # noqa: ARG001
    ) -> Tensor:
        activation[:, :, :] += injection_strength * concept_vector
        return activation

    tokens = model.to_tokens(prompt)  # shape: (batch, seq)

    with model.hooks([(f"blocks.{layer}.hook_resid_post", injection_hook)]):
        output = model.generate(
            tokens,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
        )  # shape: (batch, seq)

    result = model.to_string(output[0])
    return result
