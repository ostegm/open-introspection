"""Phase 2: Concept vector extraction using mean subtraction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from open_introspection.model import get_endoftext_token_id

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer
    from transformer_lens.hook_points import HookPoint

# Default baseline words for concept extraction (50 words per paper methodology)
# Diversified across categories: concrete objects, abstract concepts, emotions,
# actions, properties, nature, artifacts
DEFAULT_BASELINE_WORDS: list[str] = [
    # Concrete objects (15)
    "apple",
    "bicycle",
    "candle",
    "hammer",
    "mirror",
    "notebook",
    "pencil",
    "telephone",
    "umbrella",
    "wallet",
    "blanket",
    "compass",
    "lantern",
    "newspaper",
    "sculpture",
    # Nature & places (10)
    "river",
    "mountain",
    "forest",
    "island",
    "glacier",
    "volcano",
    "desert",
    "meadow",
    "canyon",
    "swamp",
    # Abstract concepts (10)
    "democracy",
    "justice",
    "freedom",
    "wisdom",
    "patience",
    "curiosity",
    "tradition",
    "progress",
    "balance",
    "chaos",
    # Actions/processes (8)
    "running",
    "building",
    "swimming",
    "teaching",
    "trading",
    "dreaming",
    "climbing",
    "writing",
    # Properties/qualities (7)
    "warmth",
    "speed",
    "heaviness",
    "brightness",
    "softness",
    "distance",
    "depth",
]


def _get_activation(
    model: HookedTransformer,
    word: str,
    layer: int,
    token_position: int = -1,
) -> Tensor:
    """Get activation for a single word at specified layer."""
    prompt = f"Tell me about {word}."
    tokens = model.to_tokens(prompt)  # shape: (batch, seq)
    _, cache = model.run_with_cache(tokens)
    # Get residual stream at specified layer and position
    activation: Tensor = cache[f"blocks.{layer}.hook_resid_post"][
        0, token_position, :
    ]  # shape: (d_model,)
    return activation


def compute_baseline_mean(
    model: HookedTransformer,
    baseline_words: list[str] | None = None,
    layer: int | None = None,
    token_position: int = -1,
    exclude_words: list[str] | None = None,
) -> Tensor:
    """
    Compute mean activation across baseline words.

    Call this once and reuse across multiple concept extractions for efficiency.

    Args:
        model: HookedTransformer model
        baseline_words: Words to use as baseline (default: DEFAULT_BASELINE_WORDS)
        layer: Layer to extract from (default: ~2/3 through model)
        token_position: Token position to extract from (-1 = last token)
        exclude_words: Words to exclude from baseline (e.g., target concepts)

    Returns:
        Mean activation tensor of shape (d_model,)
    """
    if baseline_words is None:
        baseline_words = list(DEFAULT_BASELINE_WORDS)

    if layer is None:
        layer = int(model.cfg.n_layers * 2 / 3)

    # Filter out excluded words
    original_count = len(baseline_words)
    if exclude_words:
        exclude_set = {w.lower() for w in exclude_words}
        baseline_words = [w for w in baseline_words if w.lower() not in exclude_set]
        excluded_count = original_count - len(baseline_words)
        if excluded_count > 0:
            logger.info(
                f"Excluded {excluded_count} words from baseline "
                f"({len(baseline_words)} remaining)"
            )

    logger.info(f"Computing baseline mean from {len(baseline_words)} words at layer {layer}")

    # Compute activations for all baseline words
    baseline_acts: Tensor = torch.stack(
        [_get_activation(model, w, layer, token_position) for w in baseline_words]
    )  # shape: (n_baselines, d_model)

    baseline_mean: Tensor = baseline_acts.mean(dim=0)  # shape: (d_model,)
    return baseline_mean


def extract_concept_vector(
    model: HookedTransformer,
    target_word: str,
    baseline_words: list[str] | None = None,
    layer: int | None = None,
    token_position: int = -1,
    cached_baseline_mean: Tensor | None = None,
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
        cached_baseline_mean: Pre-computed baseline mean (skips baseline computation)

    Returns:
        Concept vector as a tensor of shape (d_model,)
    """
    if layer is None:
        layer = int(model.cfg.n_layers * 2 / 3)

    # Get target activation
    target_act: Tensor = _get_activation(
        model, target_word, layer, token_position
    )  # shape: (d_model,)

    # Use cached baseline mean or compute fresh
    if cached_baseline_mean is not None:
        logger.debug(f"Using cached baseline mean for '{target_word}'")
        # Warn if target word is in default baselines (potential contamination)
        if target_word.lower() in {w.lower() for w in DEFAULT_BASELINE_WORDS}:
            logger.warning(
                f"Target word '{target_word}' is in DEFAULT_BASELINE_WORDS. "
                "If cached baseline was computed from defaults without excluding "
                "this word, the concept vector may be attenuated."
            )
        baseline_mean = cached_baseline_mean
    else:
        logger.debug(f"Computing fresh baseline for '{target_word}'")
        if baseline_words is None:
            baseline_words = list(DEFAULT_BASELINE_WORDS)

        # Auto-exclude target word from baselines
        baseline_words = [
            w for w in baseline_words if w.lower() != target_word.lower()
        ]

        baseline_acts: Tensor = torch.stack(
            [_get_activation(model, w, layer, token_position) for w in baseline_words]
        )  # shape: (n_baselines, d_model)
        baseline_mean = baseline_acts.mean(dim=0)  # shape: (d_model,)

    # Concept vector = target - baseline mean
    concept_vector: Tensor = target_act - baseline_mean  # shape: (d_model,)

    return concept_vector


def validate_concept_vector(
    model: HookedTransformer,
    concept_vector: Tensor,
    concept_name: str,  # noqa: ARG001
    layer: int,
    injection_strength: float = 2.0,
    temperature: float = 1.0,
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
        temperature: Sampling temperature (0=greedy, 1=full sampling per paper)

    Returns:
        Generated text with the concept vector injected
    """
    prompt = "Write a short sentence."

    def injection_hook(
        activation: Tensor,
        hook: HookPoint,  # noqa: ARG001
    ) -> Tensor:
        activation[:, :, :] += injection_strength * concept_vector
        return activation

    tokens = model.to_tokens(prompt)  # shape: (batch, seq)
    prompt_len = tokens.shape[1]
    eos_token_id = get_endoftext_token_id(model)

    # Use temperature=0 for greedy (do_sample=False), otherwise sample
    do_sample = temperature > 0

    with model.hooks([(f"blocks.{layer}.hook_resid_post", injection_hook)]):
        output = model.generate(
            tokens,
            max_new_tokens=50,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            stop_at_eos=True,
            eos_token_id=eos_token_id,
        )  # shape: (batch, seq)

    # Only return generated tokens (not the prompt)
    generated = output[0, prompt_len:]
    result: str = model.to_string(generated)
    return result
