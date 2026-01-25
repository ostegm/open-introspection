"""Tests for chat template utilities in model.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from open_introspection.model import (
    ChatMessages,
    format_chat_prompt,
    strip_special_tokens,
    tokenize_chat,
)

if TYPE_CHECKING:
    pass


class TestStripSpecialTokens:
    """Test special token stripping for various model formats."""

    def test_strips_qwen_im_end(self) -> None:
        text = "Hello world<|im_end|>extra stuff"
        assert strip_special_tokens(text) == "Hello world"

    def test_strips_qwen_endoftext(self) -> None:
        text = "Response here<|endoftext|>garbage"
        assert strip_special_tokens(text) == "Response here"

    def test_strips_qwen_im_start(self) -> None:
        text = "Content<|im_start|>more content"
        assert strip_special_tokens(text) == "Content"

    def test_strips_llama_eot_id(self) -> None:
        text = "Llama response<|eot_id|>trailing"
        assert strip_special_tokens(text) == "Llama response"

    def test_strips_llama_end_of_text(self) -> None:
        text = "Another response<|end_of_text|>more"
        assert strip_special_tokens(text) == "Another response"

    def test_strips_llama_header(self) -> None:
        text = "Response<|start_header_id|>assistant"
        assert strip_special_tokens(text) == "Response"

    def test_strips_mistral_inst_close(self) -> None:
        text = "Answer[/INST]more tokens"
        assert strip_special_tokens(text) == "Answer"

    def test_strips_generic_eos(self) -> None:
        text = "Some text</s>extra"
        assert strip_special_tokens(text) == "Some text"

    def test_preserves_clean_text(self) -> None:
        text = "This is clean text without special tokens"
        assert strip_special_tokens(text) == text

    def test_handles_empty_string(self) -> None:
        assert strip_special_tokens("") == ""

    def test_handles_only_special_token(self) -> None:
        assert strip_special_tokens("<|im_end|>") == ""

    def test_strips_and_trims_whitespace(self) -> None:
        text = "  Response with spaces  <|im_end|>trailing"
        assert strip_special_tokens(text) == "Response with spaces"


class TestFormatChatPrompt:
    """Test chat prompt formatting with mocked tokenizer."""

    def test_format_simple_user_message(self) -> None:
        # Create mock model with tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        messages: ChatMessages = [{"role": "user", "content": "Hello"}]
        result = format_chat_prompt(mock_model, messages, add_generation_prompt=True)

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        assert result == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    def test_format_with_system_message(self) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "formatted prompt"

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        messages: ChatMessages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        format_chat_prompt(mock_model, messages)

        # Verify the messages were passed correctly
        call_args = mock_tokenizer.apply_chat_template.call_args
        assert call_args[0][0] == messages

    def test_format_without_generation_prompt(self) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2]
        mock_tokenizer.decode.return_value = "no assistant prefix"

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        messages: ChatMessages = [{"role": "user", "content": "Test"}]
        format_chat_prompt(mock_model, messages, add_generation_prompt=False)

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )


class TestTokenizeChat:
    """Test direct tokenization of chat messages."""

    def test_tokenize_returns_token_ids(self) -> None:
        mock_tokenizer = MagicMock()
        expected_ids = [100, 200, 300, 400, 500]
        mock_tokenizer.apply_chat_template.return_value = expected_ids

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        messages: ChatMessages = [{"role": "user", "content": "Hello"}]
        result = tokenize_chat(mock_model, messages)

        assert result == expected_ids
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )

    def test_tokenize_without_generation_prompt(self) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        messages: ChatMessages = [{"role": "user", "content": "Test"}]
        tokenize_chat(mock_model, messages, add_generation_prompt=False)

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )


class TestIntrospectionPromptMessages:
    """Test that introspection prompts are properly structured."""

    def test_v1_messages_structure(self) -> None:
        from open_introspection.introspection import INTROSPECTION_MESSAGES_V1

        assert len(INTROSPECTION_MESSAGES_V1) == 1
        assert INTROSPECTION_MESSAGES_V1[0]["role"] == "user"
        assert "inject" in INTROSPECTION_MESSAGES_V1[0]["content"].lower()

    def test_v2_messages_structure(self) -> None:
        from open_introspection.introspection import INTROSPECTION_MESSAGES_V2

        assert len(INTROSPECTION_MESSAGES_V2) == 2
        assert INTROSPECTION_MESSAGES_V2[0]["role"] == "system"
        assert INTROSPECTION_MESSAGES_V2[1]["role"] == "user"

    def test_prompt_messages_dict(self) -> None:
        from open_introspection.introspection import PROMPT_MESSAGES

        assert "v1" in PROMPT_MESSAGES
        assert "v2" in PROMPT_MESSAGES
        assert len(PROMPT_MESSAGES) >= 2
