"""Unit tests for GGUF chat-template handling (Qwen3 ChatML vs generic).

Covers build_llama_cpp_prompt() with both dialects and
detect_chat_template() heuristics. Pure-Python, no model required.
"""

import sys
import os
from types import SimpleNamespace

import pytest

# Ensure src/ is importable without pulling model deps.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import gguf  # noqa: E402


MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
    {"role": "user", "content": "What is 2+2?"},
]


def test_generic_dialect():
    prompt = gguf.build_llama_cpp_prompt(MESSAGES, template="generic")
    assert "<|system|>\nYou are helpful." in prompt
    assert "<|user|>\nHello" in prompt
    assert "<|assistant|>\nHi there" in prompt
    assert prompt.endswith("<|assistant|>")


def test_qwen3_dialect():
    prompt = gguf.build_llama_cpp_prompt(MESSAGES, template="qwen3")
    assert "<|im_start|>system\nYou are helpful.<|im_end|>\n" in prompt
    assert "<|im_start|>user\nHello<|im_end|>\n" in prompt
    assert "<|im_start|>assistant\nHi there<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_detect_qwen3_by_filename():
    assert gguf.detect_chat_template("Qwen/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf") == "qwen3"
    assert gguf.detect_chat_template("Qwen3.8-Whittle-MoE-27B-A17.8B.gguf") == "qwen3"
    assert gguf.detect_chat_template("qwen3-moe:latest") == "qwen3"


def test_detect_generic_fallback():
    assert gguf.detect_chat_template("meta-llama/Llama-3.3-70B-Instruct.gguf") == "generic"
    assert gguf.detect_chat_template("") == "generic"


def test_detect_qwen2_not_matched_as_qwen3():
    # Qwen2 family uses the generic dialect; must NOT be misdetected as qwen3.
    assert gguf.detect_chat_template("Qwen/Qwen2.5-32B-A3B-Instruct-GGUF") == "generic"


def test_detect_from_metadata():
    meta = {"chat_template": "<|im_start|>...<|im_end|>"}
    assert gguf.detect_chat_template("anything.gguf", gguf_meta=meta) == "qwen3"
    meta2 = {"tokenizer.chat_template": "<|im_start|>..."}
    assert gguf.detect_chat_template("anything.gguf", gguf_meta=meta2) == "qwen3"
    assert gguf.detect_chat_template("anything.gguf", gguf_meta={}) == "generic"