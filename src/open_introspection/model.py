"""Model loading utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer


# Type alias for chat messages
ChatMessage = dict[str, str]
ChatMessages = list[ChatMessage]


def get_endoftext_token_id(model: HookedTransformer) -> int:
    """Get the <|endoftext|> token ID for Qwen models.

    Qwen's tokenizer reports <|im_end|> as eos_token_id, but the model
    actually uses <|endoftext|> to signal completion when not using chat format.
    """
    token_ids = model.tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    return int(token_ids[0])


def get_stop_token_ids(model: HookedTransformer) -> list[int]:
    """Get all stop token IDs for Qwen chat format.

    Returns both <|im_end|> (chat turn end) and <|endoftext|> (sequence end).
    """
    tokenizer = model.tokenizer
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    endoftext = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    return [int(im_end), int(endoftext)]


def strip_special_tokens(text: str) -> str:
    """Remove common special tokens from generated text.

    Works with Qwen, Llama, Mistral, and other common chat models.
    """
    import re

    # Remove special tokens and everything after them
    # Covers: Qwen (<|im_*|>, <|endoftext|>), Llama (<|eot_id|>, <|end_of_text|>),
    # Mistral ([INST], [/INST]), and generic markers
    pattern = (
        r"<\|im_end\|>.*|<\|endoftext\|>.*|<\|im_start\|>.*|"
        r"<\|eot_id\|>.*|<\|end_of_text\|>.*|<\|start_header_id\|>.*|"
        r"\[/INST\].*|\</s\>.*"
    )
    text = re.split(pattern, text)[0]
    return text.strip()


def tokenize_chat(
    model: HookedTransformer,
    messages: ChatMessages,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Tokenize messages using the model's chat template.

    This is the preferred way to prepare prompts for chat models, as it
    automatically handles model-specific formatting (Qwen, Llama, Mistral, etc.)
    and returns token IDs directly without decode/re-encode overhead.

    Args:
        model: HookedTransformer model with tokenizer
        messages: List of message dicts with 'role' and 'content' keys
            Roles are typically: 'system', 'user', 'assistant'
        add_generation_prompt: Whether to add the assistant turn prefix
            (e.g., "<|im_start|>assistant\n" for Qwen)

    Returns:
        List of token IDs
    """
    tokenizer = model.tokenizer
    token_ids: list[int] = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )
    return token_ids


def format_chat_prompt(
    model: HookedTransformer,
    messages: ChatMessages,
    add_generation_prompt: bool = True,
) -> str:
    """Format messages as a string using the model's chat template.

    Convenience wrapper around tokenize_chat for when you need the string
    representation. For generation, prefer tokenize_chat to avoid re-tokenization.

    Args:
        model: HookedTransformer model with tokenizer
        messages: List of message dicts with 'role' and 'content' keys
        add_generation_prompt: Whether to add the assistant turn prefix

    Returns:
        Formatted prompt string
    """
    token_ids = tokenize_chat(model, messages, add_generation_prompt)
    result: str = model.tokenizer.decode(token_ids)
    return result


def load_model(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
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
        "dtype": dtype,
        "device": device if device != "mps" else "cpu",
    }

    if load_in_4bit:
        kwargs["load_in_4bit"] = True
        # 4-bit quantization needs device_map instead of device
        del kwargs["device"]
        kwargs["device_map"] = device if device != "mps" else None

    # Use no_processing variant to avoid device mismatch during weight folding
    # This is recommended for reduced precision (bfloat16/float16)
    if dtype != torch.float32 and not load_in_4bit:
        model: HookedTransformer = HookedTransformer.from_pretrained_no_processing(
            model_name, **kwargs
        )
    else:
        model = HookedTransformer.from_pretrained(model_name, **kwargs)

    if device == "mps":
        print(f"Moving model to device:  {device}")
        model = model.to("mps")

    # Disable gradients for inference - saves memory and improves speed
    torch.set_grad_enabled(False)
    model.eval()

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
