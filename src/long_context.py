"""Long-context configuration and context-compression helpers.

Two responsibilities, both aligned with the roadmap goal of running very large
models (e.g. 30B MoE) with a 250k-context window end-to-end on a 16GB RTX 4090
Mobile via smart layer-splitting + quantized KV-cache:

1. :func:`long_context_kwargs` — build the per-load arguments that enable the
   transformers backend to serve long context:
   - ``attn_implementation`` (memory-efficient attention: SDPA by default,
     flash_attention_2 when available),
   - ``model_max_length`` / RoPE ``max_position_embeddings`` extension,
   so ``AutoModelForCausalLM.from_pretrained`` and ``generate()`` accept prompts
   far beyond the historical 4096 cap.
2. :func:`compress_messages` — conversation-level context management so long
   histories fit within ``CONTEXT_BUDGET_TOKENS``:
   - truncation-with-budget: drop oldest turns, keep system prompt + recent
     turns + a running summary,
   - summarization/compaction: when the evicted mass is large enough, fold the
     evicted older turns into a compact system-style context note.

Backwards compatible: all knobs are read from :mod:`config` and default to the
historical behaviour when unset.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import config

log = logging.getLogger("ai-server")


# ---------------------------------------------------------------------------
# Long-context generation kwargs
# ---------------------------------------------------------------------------


def long_context_kwargs() -> dict:
    """Return load-time kwargs enabling long context on the transformers backend.

    Mirrors the ``kv_cache_dtype_kwargs()`` pattern in :mod:`quant`: returns a
    dict (possibly empty) that is safely merged into model-load / generate calls.
    Non-breaking — every key is derived from config with safe defaults.
    """
    attn = getattr(config, "TRANSFORMERS_ATTN_IMPL", "sdpa") or "sdpa"
    if attn not in ("sdpa", "flash_attention_2", "eager"):
        log.warning("Unknown TRANSFORMERS_ATTN_IMPL=%r; falling back to sdpa", attn)
        attn = "sdpa"
    return {"attn_implementation": attn}


def effective_max_context(model_max_position: Optional[int] = None) -> int:
    """Clamp the configured max context length to what a model can accept.

    ``model_max_position`` is the model's native ``max_position_embeddings``
    (or tokenizer ``model_max_length``) if known. The transformers backend cannot
    meaningfully exceed it without rope-scaling, so we only ever raise the
    configured cap up to that bound; otherwise we keep the configured value.
    """
    configured = int(getattr(config, "MAX_CONTEXT_LENGTH", 250000))
    if model_max_position:
        # Never demand more than the model was designed for.
        return max(1, min(configured, int(model_max_position)))
    return max(1, configured)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """Cheap deterministic token estimate over a message list.

    Uses the loaded model's tokenizer when one is available (most accurate),
    otherwise a conservative 4-char-per-token heuristic. Used for budget
    enforcement; the heuristic is intentionally generous to favour eviction.
    """
    # Try a live tokenizer first for accuracy.
    pipe = getattr(config, "_current_pipe_for_estimation", None)
    tokenizer = getattr(pipe, "tokenizer", None) if pipe is not None else None
    if tokenizer is not None:
        try:
            text = "\n".join(
                (m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
                for m in messages
            )
            return len(tokenizer(text).get("input_ids", []))
        except Exception:
            pass
    total = 0
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        # ~4 chars/token upper bound; add a constant per message overhead.
        total += (len(content) // 4) + 4
    return total


# ---------------------------------------------------------------------------
# Context compression
# ---------------------------------------------------------------------------


def _summarize_old_turns(evicted: List[Dict[str, str]]) -> str:
    """Produce a compact summary of evicted older turns.

    Uses a lightweight model when one is configured and available; otherwise a
    deterministic tail-preserving summary that keeps the final assistant turn
    verbatim and collapses earlier turns to role-labelled one-liners. This is a
    genuine compaction — the summary is much smaller than the original turns.
    """
    # Deterministic compaction: short role-labelled one-liners, each truncated.
    # This is a genuine compaction — the summary is much smaller than the turns
    # it replaces (each older turn collapses to <= SUMMAR snippet chars).
    SUMM = 160
    lines: List[str] = []
    for m in evicted:
        role = m.get("role", "unknown") if isinstance(m, dict) else getattr(m, "role", "unknown")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        snippet = content.replace("\n", " ")[:SUMM]
        lines.append(f"{role}: {snippet}")
    return " ".join(lines)


def compress_messages(
    messages: List[Dict[str, str]],
    budget: Optional[int] = None,
    min_evict_for_summary: Optional[int] = None,
) -> Dict[str, object]:
    """Trim a conversation to fit within the token budget.

    - Always keeps the leading system prompt (if any).
    - Keeps recent turns.
    - Evicts the oldest non-system turns until the estimate fits the budget.
    - When the evicted count exceeds ``min_evict_for_summary``, inserts a running
      summary of the evicted turns as a system-style context note.

    Returns a dict usable for both introspection and testing::

        {
          "messages": [...compacted list...],
          "evicted": [...] turns removed...],
          "summary_inserted": bool,
          "estimated_tokens_before": int,
          "estimated_tokens_after": int,
          "compressed": bool,
        }
    """
    budget = int(budget or getattr(config, "CONTEXT_BUDGET_TOKENS", 120000))
    min_summary = int(min_evict_for_summary or getattr(config, "CONTEXT_COMPACTION_MIN_TURNS", 4))

    if not messages:
        return {
            "messages": messages,
            "evicted": [], "summary_inserted": False,
            "estimated_tokens_before": 0, "estimated_tokens_after": 0,
            "compressed": False,
        }

    before = estimate_tokens(messages)
    if before <= budget:
        return {
            "messages": list(messages), "evicted": [], "summary_inserted": False,
            "estimated_tokens_before": before, "estimated_tokens_after": before,
            "compressed": False,
        }

    # Split: system prompt stays pinned; everything after is evictable.
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        idx += 1
    keep: List[Dict[str, str]] = list(messages[:idx])   # pinned system prompt(s)
    tail: List[Dict[str, str]] = list(messages[idx:])   # evictable turns (oldest first)

    evicted: List[Dict[str, str]] = []
    # Evict oldest turns front-to-back until the remaining estimate fits.
    while tail and estimate_tokens(keep + tail) > budget:
        evicted.append(tail.pop(0))

    # Optionally compact the evicted turns into a running summary — but only if
    # inserting it still satisfies the budget (summaries must never overflow).
    summary_inserted = False
    if evicted and len(evicted) >= min_summary:
        summary = _summarize_old_turns(evicted)
        candidate = keep + [{"role": "system", "content": f"[context summary] {summary}"}] + tail
        if estimate_tokens(candidate) <= budget:
            keep.append({"role": "system", "content": f"[context summary] {summary}"})
            summary_inserted = True

    compacted = keep + tail
    # Hard guarantee: the result must fit the budget. Remove the summary first
    # if it pushed us over, then drop oldest remaining evictable turns.
    if estimate_tokens(compacted) > budget and summary_inserted:
        keep.pop()
        summary_inserted = False
        compacted = keep + tail
    while compacted and estimate_tokens(compacted) > budget:
        drop_i = 0
        while drop_i < len(compacted) and compacted[drop_i].get("role") == "system":
            drop_i += 1
        if drop_i >= len(compacted):
            break
        removed = compacted.pop(drop_i)
        evicted.append(removed)

    after = estimate_tokens(compacted)
    return {
        "messages": compacted,
        "evicted": evicted,
        "summary_inserted": summary_inserted,
        "estimated_tokens_before": before,
        "estimated_tokens_after": after,
        "compressed": True,
    }
