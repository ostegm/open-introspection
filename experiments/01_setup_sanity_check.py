#!/usr/bin/env python
"""Phase 1: Setup and sanity checks.

Verify that:
1. Model loads correctly
2. Activations are accessible at all layers
3. Memory usage is manageable
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from open_introspection.model import load_model, verify_activation_access

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def main() -> None:
    """Run sanity checks for model loading and activation access."""
    print("=" * 60)
    print("Phase 1: Setup & Sanity Checks")
    print("=" * 60)

    # Detect device
    device: str
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("Using CPU (this will be slow)")

    # Load model - start with smaller model for testing
    print("\nLoading model...")
    model: HookedTransformer = load_model(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device=device,
        dtype=torch.bfloat16,
    )
    print(f"Model loaded: {model.cfg.model_name}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hidden dim: {model.cfg.d_model}")
    print(f"  Heads: {model.cfg.n_heads}")

    # Verify activation access
    print("\nVerifying activation access...")
    _ = verify_activation_access(model)

    # Test generation
    print("\nTesting generation...")
    prompt: str = "Hello! Can you explain what machine learning is in one sentence?"
    tokens = model.to_tokens(prompt)  # shape: (batch, seq)
    prompt_len = tokens.shape[1]

    # Qwen uses <|endoftext|> as actual stop token, not <|im_end|>
    endoftext_id = model.tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    print(f"  Using <|endoftext|> token id: {endoftext_id}")

    output = model.generate(
        tokens,
        max_new_tokens=50,
        temperature=0.7,
        stop_at_eos=True,
        eos_token_id=endoftext_id,
    )  # shape: (batch, seq)

    # Only show generated tokens (not the prompt)
    generated_tokens = output[0, prompt_len:]
    response = model.to_string(generated_tokens)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    print("\n" + "=" * 60)
    print("Sanity checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
