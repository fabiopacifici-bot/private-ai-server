"""Unit tests for GGUF inference safety + throughput wiring.

Covers:
- gguf.infer_lock exists and serializes the shared Llama instance (the CUDA
  crash fix: concurrent calls on one Llama object are not thread-safe).
- load_gguf() passes n_batch from GGUF_N_BATCH env (benchmark-derived default
  1024: 126 -> 146 -> 137 t/s at 512/1024/2048 on Qwen3-30B-A3B Q2_K).
- generate_response()'s gguf branch holds the lock across generation and
  releases it afterwards.

Pure-Python: a fake Llama records invocation; no model is loaded.
"""

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import gguf  # noqa: E402
import state  # noqa: E402
import config  # noqa: E402
import model_cache as mcache  # noqa: E402
from routes.chat import generate_response  # noqa: E402
from schemas import ChatRequest  # noqa: E402


# ---------------------------------------------------------------------------
# Lock presence
# ---------------------------------------------------------------------------


def test_infer_lock_exists_and_is_held_not_busy():
    """The lock is a real threading.Lock and starts free."""
    assert isinstance(gguf.infer_lock, threading.Lock)
    assert not gguf.infer_lock.locked()


def test_acquire_release_roundtrip():
    """Acquire/release behaves like a normal lock (used by both paths)."""
    gguf.infer_lock.acquire()
    try:
        assert gguf.infer_lock.locked()
    finally:
        gguf.infer_lock.release()
    assert not gguf.infer_lock.locked()


# ---------------------------------------------------------------------------
# load_gguf() env wiring
# ---------------------------------------------------------------------------


class FakeLlama:
    """Records constructor kwargs like the real Llama would consume them."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _patch_llama(monkeypatch, tmp_path, fake_cls=FakeLlama):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    monkeypatch.setattr(config, "llama_cpp_available", True)
    monkeypatch.setattr(config, "LlamaCpp", fake_cls)
    captured = {}

    def _fake_cache(name, backend, instance, extra_meta=None):
        captured["instance"] = instance
        captured["extra_meta"] = extra_meta or {}

    monkeypatch.setattr(mcache, "cache_model", _fake_cache)
    return str(gguf_path), captured


def test_load_gguf_default_n_batch_1024(monkeypatch, tmp_path):
    monkeypatch.delenv("GGUF_N_BATCH", raising=False)
    gguf_path, captured = _patch_llama(monkeypatch, tmp_path)
    gguf.load_gguf("test-model", gguf_path)
    assert captured["instance"].kwargs["n_batch"] == 1024
    assert captured["instance"].kwargs["n_gpu_layers"] == -1
    assert captured["extra_meta"]["n_batch"] == 1024


def test_load_gguf_n_batch_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GGUF_N_BATCH", "2048")
    gguf_path, captured = _patch_llama(monkeypatch, tmp_path)
    gguf.load_gguf("test-model", gguf_path)
    assert captured["instance"].kwargs["n_batch"] == 2048


def test_load_gguf_gpu_layers_env(monkeypatch, tmp_path):
    monkeypatch.delenv("GGUF_N_GPU_LAYERS", raising=False)
    monkeypatch.setenv("GGUF_N_GPU_LAYERS", "40")
    gguf_path, captured = _patch_llama(monkeypatch, tmp_path)
    gguf.load_gguf("test-model", gguf_path)
    assert captured["instance"].kwargs["n_gpu_layers"] == 40


# ---------------------------------------------------------------------------
# generate_response() gguf path: lock held across generation
# ---------------------------------------------------------------------------


class LockCheckingLlama:
    """Fake Llama that fails if the inference lock is not held when called."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, prompt, **gen_kwargs):
        assert gguf.infer_lock.locked(), (
            "GGUF generation ran without holding infer_lock"
        )
        return {
            "choices": [{"text": " hello world "}],
            "usage": {"completion_tokens": 4},
        }


@pytest.fixture
def gguf_model_state(monkeypatch):
    llm = LockCheckingLlama()
    monkeypatch.setattr(state, "current_model", llm)
    monkeypatch.setattr(state, "current_model_name", "Qwen3-30B-A3B.Q2_K.gguf")
    monkeypatch.setattr(state, "model_meta", {
        "Qwen3-30B-A3B.Q2_K.gguf": {"backend": "gguf_llama_cpp"},
    })
    return llm


def test_generate_response_gguf_holds_and_releases_lock(monkeypatch, gguf_model_state):
    request = ChatRequest(
        model="Qwen3-30B-A3B.Q2_K.gguf",
        messages=[{"role": "user", "content": "hi"}],
    )
    resp = generate_response(
        request, "Qwen3-30B-A3B.Q2_K.gguf", "User: hi\nAssistant: ",
        "gguf_llama_cpp",
    )
    assert resp["message"]["content"] == "hello world"
    assert not gguf.infer_lock.locked(), "lock must be released after generation"


def test_generate_response_gguf_qwen3_stop_and_template(monkeypatch, gguf_model_state):
    """Qwen3-family models must get ChatML stop tokens."""
    request = ChatRequest(
        model="Qwen3-30B-A3B.Q2_K.gguf",
        messages=[{"role": "user", "content": "hi"}],
    )
    resp = generate_response(
        request, "Qwen3-30B-A3B.Q2_K.gguf", "User: hi\nAssistant: ",
        "gguf_llama_cpp",
    )
    assert resp["message"]["content"] == "hello world"