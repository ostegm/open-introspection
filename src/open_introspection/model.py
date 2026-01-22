"""Model loading utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer


def load_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
    load_in_4bit: bool = False,
) -> HookedTransformer:
    """
    Load a model with TransformerLens for activation access.

    Args:
        model_name: HuggingFace model name
        device: Device to load on ("auto", "mps", "cuda", "cpu")
        dtype: Torch dtype for model weights
        load_in_4bit: Whether to use 4-bit quantization (requires bitsandbytes)

    Returns:
        HookedTransformer model ready for activation analysis
    """
    from transformer_lens import HookedTransformer

    # Determine device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device if device != "mps" else None,
    }

    if load_in_4bit:
        kwargs["load_in_4bit"] = True

    model: HookedTransformer = HookedTransformer.from_pretrained(model_name, **kwargs)

    if device == "mps":
        model = model.to("mps")

    return model


def verify_activation_access(
    model: HookedTransformer, prompt: str = "Hello, how are you?"
) -> dict[int, tuple[int, ...]]:
    """
    Verify that we can access activations at different layers.

    Args:
        model: HookedTransformer model to verify
        prompt: Test prompt to run through the model

    Returns:
        Dict mapping layer index to activation shape tuple
    """
    tokens = model.to_tokens(prompt)  # shape: (batch, seq)
    _, cache = model.run_with_cache(tokens)

    n_layers: int = model.cfg.n_layers
    test_layers: list[int] = [0, n_layers // 2, n_layers - 1]

    results: dict[int, tuple[int, ...]] = {}
    for layer in test_layers:
        resid: Tensor = cache[f"blocks.{layer}.hook_resid_post"]
        results[layer] = tuple(resid.shape)
        print(f"Layer {layer} residual shape: {resid.shape}")

    return results
