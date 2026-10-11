"""
Unit tests for long_context.py — 250k-context configuration and
conversation context compression.

These tests are fully isolated: they import ``config`` and ``long_context``
directly with the repo's ``conftest`` path setup, use no network, no GPU and no
dev database. They only exercise deterministic pure functions.
"""

import pytest

import config
import long_context as lc


class TestLongContextKwargs:
    def test_default_attn_is_sdpa(self):
        kwargs = lc.long_context_kwargs()
        assert kwargs.get("attn_implementation") == "sdpa"

    def test_attn_flash_when_requested(self, monkeypatch):
        monkeypatch.setattr(config, "TRANSFORMERS_ATTN_IMPL", "flash_attention_2")
        assert lc.long_context_kwargs()["attn_implementation"] == "flash_attention_2"

    def test_unknown_attn_falls_back_to_sdpa(self, monkeypatch):
        monkeypatch.setattr(config, "TRANSFORMERS_ATTN_IMPL", "bogus")
        assert lc.long_context_kwargs()["attn_implementation"] == "sdpa"


class TestEffectiveMaxContext:
    def test_default_uses_configured_250k(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONTEXT_LENGTH", 250000)
        assert lc.effective_max_context(model_max_position=None) == 250000

    def test_clamped_to_model_native_bound(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONTEXT_LENGTH", 250000)
        # A model designed for 131072 cannot be honestly pushed past that
        # without rope-scaling; we clamp to the native bound.
        assert lc.effective_max_context(model_max_position=131072) == 131072

    def test_never_below_one(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONTEXT_LENGTH", -5)
        assert lc.effective_max_context(model_max_position=None) == 1


class TestEstimateTokens:
    def test_empty_is_zero(self):
        assert lc.estimate_tokens([]) == 0

    def test_heuristic_scales_with_content(self):
        short = lc.estimate_tokens([{"role": "user", "content": "hi"}])
        long_ = lc.estimate_tokens([{"role": "user", "content": "x" * 4000}])
        assert long_ > short


class TestCompressMessages:
    def test_no_compression_under_budget(self, monkeypatch):
        monkeypatch.setattr(config, "CONTEXT_BUDGET_TOKENS", 120000)
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        out = lc.compress_messages(msgs)
        assert out["compressed"] is False
        assert out["summary_inserted"] is False
        assert len(out["messages"]) == 2

    def test_old_turn_eviction(self, monkeypatch):
        monkeypatch.setattr(config, "CONTEXT_BUDGET_TOKENS", 50)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "A" * 1000},   # old
            {"role": "assistant", "content": "B" * 1000},  # old
            {"role": "user", "content": "recent question"},
        ]
        out = lc.compress_messages(msgs)
        assert out["compressed"] is True
        assert len(out["evicted"]) >= 1
        # System prompt is always pinned.
        assert out["messages"][0]["role"] == "system"
        # Recent turn survives.
        contents = [m["content"] for m in out["messages"]]
        assert any("recent question" in c for c in contents)

    def test_summary_inserted_when_evicted_substantial(self, monkeypatch):
        # Budget large enough that a compact summary of the evicted turns fits,
        # but small enough that several old turns must be evicted.
        monkeypatch.setattr(config, "CONTEXT_BUDGET_TOKENS", 1000)
        monkeypatch.setattr(config, "CONTEXT_COMPACTION_MIN_TURNS", 3)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old1 " + "x" * 1000},
            {"role": "assistant", "content": "old2 " + "y" * 1000},
            {"role": "user", "content": "old3 " + "z" * 1000},
            {"role": "user", "content": "old4 " + "w" * 1000},
            {"role": "user", "content": "old5 " + "v" * 1000},
            {"role": "user", "content": "old6 " + "u" * 1000},
            {"role": "user", "content": "recent question"},
        ]
        out = lc.compress_messages(msgs)
        assert out["compressed"] is True
        assert len(out["evicted"]) >= 3
        assert out["summary_inserted"] is True
        # A system-style summary note is present and the whole result fits budget.
        summaries = [m for m in out["messages"]
                     if m["role"] == "system" and m["content"].startswith("[context summary]")]
        assert len(summaries) == 1
        assert "old1" in summaries[0]["content"] or "old2" in summaries[0]["content"]
        assert out["estimated_tokens_after"] <= 1000

    def test_budget_enforcement_after_compression(self, monkeypatch):
        monkeypatch.setattr(config, "CONTEXT_BUDGET_TOKENS", 60)
        msgs = [{"role": "user", "content": c * 300} for c in "abcdefghij"]
        out = lc.compress_messages(msgs)
        assert out["compressed"] is True
        assert out["estimated_tokens_after"] <= 60

    def test_min_turns_gate_no_summary(self, monkeypatch):
        monkeypatch.setattr(config, "CONTEXT_BUDGET_TOKENS", 30)
        monkeypatch.setattr(config, "CONTEXT_COMPACTION_MIN_TURNS", 10)
        msgs = [
            {"role": "user", "content": "only" + "x" * 800},
            {"role": "user", "content": "recent"},
        ]
        out = lc.compress_messages(msgs)
        assert out["compressed"] is True
        # Fewer than 10 turns evicted -> no summary, plain eviction.
        assert out["summary_inserted"] is False


class TestLoaderHelpers:
    def test_apply_long_context_extends_and_scales(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONTEXT_LENGTH", 250000)
        from loader import _apply_long_context

        class FakeCfg:
            def __init__(self):
                self.max_position_embeddings = 4096
                self.rope_scaling = None
                self.rope_theta = 10000.0
                self.model_type = "test"
                self.__dict__ = vars(self)

        cfg = FakeCfg()
        _apply_long_context(cfg)
        assert cfg.max_position_embeddings == 250000
        assert cfg.rope_scaling is not None
        assert cfg.rope_scaling["type"] == "linear"

    def test_apply_long_context_never_exceeds_native(self, monkeypatch):
        # When the configured target is below native, we do NOT shrink the model
        # (never demand less than it was designed for).
        monkeypatch.setattr(config, "MAX_CONTEXT_LENGTH", 4096)
        from loader import _apply_long_context

        class FakeCfg:
            def __init__(self):
                self.max_position_embeddings = 131072  # native cap
                self.rope_scaling = None
                self.model_type = "test"
                self.__dict__ = vars(self)

        cfg = FakeCfg()
        _apply_long_context(cfg)
        assert cfg.max_position_embeddings == 131072
        assert cfg.rope_scaling is None
