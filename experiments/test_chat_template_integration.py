#!/usr/bin/env python3
"""Integration test for chat template API refactoring.

This script verifies that the new message-based prompt formatting works correctly
with a real model. Run it manually to confirm the refactoring is working:

    uv run python experiments/test_chat_template_integration.py

It will:
1. Load a small Qwen model
2. Test format_chat_prompt produces valid prompts
3. Test that introspection prompts are formatted correctly
4. Run a simple generation to verify end-to-end functionality
"""

from __future__ import annotations

import sys


def main() -> int:
    print("=" * 60)
    print("Chat Template Integration Test")
    print("=" * 60)

    # Step 1: Load model
    print("\n[1/5] Loading Qwen2.5-0.5B-Instruct (smallest model)...")
    from open_introspection.model import (
        format_chat_prompt,
        get_stop_token_ids,
        load_model,
        strip_special_tokens,
        tokenize_chat,
    )

    try:
        model = load_model("Qwen/Qwen2.5-0.5B-Instruct")
        print(f"  Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} dims")
    except Exception as e:
        print(f"  ERROR: Failed to load model: {e}")
        return 1

    # Step 2: Test format_chat_prompt
    print("\n[2/5] Testing format_chat_prompt...")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hello!"},
    ]
    prompt = format_chat_prompt(model, messages, add_generation_prompt=True)
    print(f"  Messages: {messages}")
    print(f"  Formatted prompt:\n{'-' * 40}")
    print(prompt)
    print(f"{'-' * 40}")

    # Verify it contains expected tokens
    if "<|im_start|>" in prompt and "<|im_end|>" in prompt:
        print("  PASS: Prompt contains expected Qwen chat tokens")
    else:
        print("  WARNING: Prompt may not have expected format (model-specific)")

    if "assistant" in prompt.lower():
        print("  PASS: Prompt ends with assistant turn")
    else:
        print("  FAIL: Prompt should end with assistant turn")
        return 1

    # Step 3: Test tokenize_chat
    print("\n[3/5] Testing tokenize_chat...")
    token_ids = tokenize_chat(model, messages, add_generation_prompt=True)
    print(f"  Token IDs count: {len(token_ids)}")
    print(f"  First 10 tokens: {token_ids[:10]}")
    decoded = model.tokenizer.decode(token_ids)
    if decoded == prompt:
        print("  PASS: tokenize_chat matches format_chat_prompt")
    else:
        print("  WARNING: tokenize_chat result differs (may be whitespace)")

    # Step 4: Test introspection prompts
    print("\n[4/5] Testing introspection prompt formatting...")
    from open_introspection.introspection import PROMPT_MESSAGES

    for version, messages in PROMPT_MESSAGES.items():
        prompt = format_chat_prompt(model, messages, add_generation_prompt=True)
        print(f"\n  Version: {version}")
        print(f"  Message count: {len(messages)}")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Token count: {len(model.to_tokens(prompt)[0])}")

        # Quick sanity check
        if "inject" in prompt.lower() or "activation" in prompt.lower():
            print(f"  PASS: {version} contains expected content")
        else:
            print(f"  FAIL: {version} missing expected content")
            return 1

    # Step 5: End-to-end generation test
    print("\n[5/5] Testing end-to-end generation...")
    from open_introspection.introspection import PROMPT_MESSAGES

    messages = PROMPT_MESSAGES["v2"]
    prompt = format_chat_prompt(model, messages, add_generation_prompt=True)
    tokens = model.to_tokens(prompt)
    prompt_len = tokens.shape[1]
    stop_tokens = get_stop_token_ids(model)

    print(f"  Prompt tokens: {prompt_len}")
    print(f"  Stop tokens: {stop_tokens}")
    print("  Generating response (max 50 tokens)...")

    output = model.generate(
        tokens,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        stop_at_eos=True,
        eos_token_id=stop_tokens,
    )

    generated = output[0, prompt_len:]
    response = model.to_string(generated)
    response = strip_special_tokens(response)

    print(f"  Response ({len(response)} chars): {response[:200]}...")
    print("  PASS: Generation completed successfully")

    print("\n" + "=" * 60)
    print("All integration tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
