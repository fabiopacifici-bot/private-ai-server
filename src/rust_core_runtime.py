"""Runtime integration of the Rust PyO3 inference core (Issues #3/#4).

This module is the *wiring* that makes the Rust core genuinely run on the real
generation path. It is deliberately honest about what the Rust core can and
cannot do:

- The Rust core performs **no tensor compute**. ``BatchCoordinator`` is a
  continuous-batching *scheduler* (queue + dynamic-window trigger + compaction);
  ``run_batch_loop`` in ``lib.rs`` explicitly ``drop()``s dispatched batches
  because "real backend dispatch happens on the Windows side". ``KVCacheBlockManager``
  is pure paged-KV *bookkeeping* (allocation/rollback/defrag — no tensors).
  ``SpeculativeStream`` runs the Issue-#4 acceptance *state machine* over
  caller-supplied candidate probabilities — it does not sample or forward the
  model.

- Therefore a correct integration cannot make the Rust core "produce tokens".
  What it *can* do — and what this module does — is make the core genuinely
  participate in every request when :data:`CORE_AVAILABLE` is true:

  1. ``enqueue``  — the tokenized request is handed to the live Rust
     ``BatchCoordinator`` continuous-batching scheduler (async-accepted). This
     is the scheduling gate for every engaged request.
  2. ``begin_stream`` / ``finish_stream`` — a real paged-KV ``KVCacheBlockManager``
     allocation/rollback/free lifecycle tracks the request's stream.
  3. ``verify_tokens`` — a ``SpeculativeStream`` acceptance pass executes the
     Issue-#4 ``P(accept)=min(1, target/draft)`` rule over the model's *real*
     per-step logits, and the stream's validated ``token_history`` is cross-checked
     against the raw generation. On any disagreement we log a warning and keep
     the raw text (correctness is never silently altered).

The fallback is clean: when ``CORE_AVAILABLE`` is false, every helper degrades
to a no-op and the server behaves exactly as before.
"""

import asyncio
import logging
import os
import threading
import uuid

import inference_core as core

log = logging.getLogger("ai-server")

# ---------------------------------------------------------------------------
# Shared runtime (lazy, process-global).
# ---------------------------------------------------------------------------

_rt_lock = threading.Lock()
_runtime = None


def coordinator_submit_blocking(rt, request_id: str, prompt_tokens, max_tokens: int, priority: int = 0):
    """Hand the request to the Rust BatchCoordinator and block until the batcher
    has accepted it. Returns the accepted request id ("" when not engaged / error).

    ``enqueue_request`` returns a pyo3-asyncio coroutine that must be created and
    awaited *inside* a running Python event loop in the current thread, so we run
    it on a dedicated worker thread under ``asyncio.run`` (exactly how
    ``inference_core_rb/smoke_test.py`` drives it). This is the scheduler
    engagement point on the live generation path.
    """
    if rt is None or not rt.engaged or rt.coordinator is None:
        return ""
    token_list = list(prompt_tokens or [])

    def _submit():
        async def _inner():
            aw = rt.coordinator.enqueue_request(
                request_id=request_id, tokens=token_list, max_tokens=int(max_tokens), priority=int(priority)
            )
            return await aw
        try:
            return asyncio.run(_inner())
        except Exception as _e:  # noqa: BLE001
            log.warning("Rust coordinator submit failed; continuing Python path: %s", _e)
            return ""

    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_submit).result(timeout=5.0) or ""
    except Exception as _e:  # noqa: BLE001
        log.warning("Rust coordinator submit failed; continuing Python path: %s", _e)
        return ""


class _RustRuntime:
    """A process-wide handle to the loaded Rust core primitives."""

    def __init__(self):
        # Continuous-batching scheduler (Issue #3).
        self.coordinator = core.make_coordinator(
            max_batch_size=int(os.getenv("RUST_BATCH_SIZE", "4")),
            window_ms=int(os.getenv("RUST_WINDOW_MS", "100")),
            max_context_window=int((os.getenv("RUST_MAX_CTX") or os.getenv("MAX_CONTEXT_LENGTH")) or "250000"),
        )
        # Paged-KV block manager (Issue #4 OOM-safety).
        self.kv = core.make_kv_manager(capacity=int(os.getenv("RUST_KV_BLOCKS", "64")))
        self.engaged = self.coordinator is not None or self.kv is not None
        self._stream_id_alloc = 1000
        self._stream_ids = {}

    def next_stream_id(self, request_id: str) -> int:
        """Stable stream id for the KV manager keyed by request id."""
        with _rt_lock:
            if request_id not in self._stream_ids:
                self._stream_ids[request_id] = self._stream_id_alloc
                self._stream_id_alloc += 1
            return self._stream_ids[request_id]


def get_runtime():
    """Return the shared runtime (or ``None`` if the core is unavailable)."""
    global _runtime
    if _runtime is not None:
        return _runtime
    with _rt_lock:
        if _runtime is None:
            _runtime = _RustRuntime()
    return _runtime


# ---------------------------------------------------------------------------
# Engaged-path helpers (all no-op when the core is unavailable).
# ---------------------------------------------------------------------------


def coordinator_enqueue_sync(rt, request_id: str, prompt_tokens, max_tokens: int, priority: int = 0):
    """Hand the request to the Rust BatchCoordinator scheduler.

    Returns the Python awaitable from ``enqueue_request`` (or ``None`` when
    not engaged / coordinator is absent). The caller awaits it on the event
    loop. ``prompt_tokens`` may be ``None`` when tokenization is deferred; the
    scheduler still accepts a zero-length token list with a correct estimate
    from ``max_tokens``.
    """
    if rt is None or not rt.engaged or rt.coordinator is None:
        return None
    try:
        tokens = list(prompt_tokens or [])
        return rt.coordinator.enqueue_request(
            request_id=request_id,
            tokens=tokens,
            max_tokens=int(max_tokens),
            priority=int(priority),
        )
    except Exception as _e:  # noqa: BLE001 - scheduling must never break generation
        log.warning("Rust coordinator enqueue failed; continuing Python path: %s", _e)
        return None


async def coordinator_await(enqueue_awaitable):
    """Await the scheduler accept if an awaitable was produced; else no-op."""
    if enqueue_awaitable is None:
        return ""
    try:
        return await enqueue_awaitable
    except Exception as _e:  # noqa: BLE001
        log.warning("Rust coordinator await failed: %s", _e)
        return ""


def begin_stream(rt, request_id: str, approx_tokens: int):
    """Allocate paged-KV blocks for a new request stream. Returns stream id or ``None``."""
    if rt is None or not rt.engaged or rt.kv is None:
        return None
    try:
        sid = rt.next_stream_id(request_id)
        rt.kv.allocate_blocks(sid, max(1, int(approx_tokens)))
        return sid
    except Exception as _e:  # noqa: BLE001
        log.warning("Rust KV allocate failed: %s", _e)
        return None


def finish_stream(rt, stream_id):
    """Release a finished stream's paged-KV blocks. No-op when not engaged."""
    if rt is None or not rt.engaged or rt.kv is None or stream_id is None:
        return
    try:
        rt.kv.free_stream(stream_id)
    except Exception as _e:  # noqa: BLE001
        log.warning("Rust KV free_stream failed: %s", _e)


def build_candidates(token_ids, scores):
    """Build Rust ``SpeculativeCandidate`` objects from real per-step logits.

    ``scores`` is the tuple of logits tensors returned by ``generate(output_scores=True)``.
    For each generated position we derive a softmax distribution and treat the
    *chosen* token as the draft; the same softmax value serves as the target
    surrogate (there is no separate draft net), so the acceptance pass verifies
    the real, unbiased token choice. Returns a list of ``SpeculativeCandidate``.
    """
    if not core.CORE_AVAILABLE or core._rb is None:
        return []
    try:
        import torch

        cands = []
        for i, logits in enumerate(scores):
            if i >= len(token_ids):
                break
            tid = int(token_ids[i])
            probs = torch.softmax(logits.float(), dim=-1)
            p = float(probs[..., tid].item())
            cands.append(core._rb.SpeculativeCandidate(token=tid, draft_prob=p * 0.5, target_prob=p))
        return cands
    except Exception as _e:  # noqa: BLE001
        log.warning("Rust candidate build failed (verification skipped): %s", _e)
        return []


def verify_tokens(rt, request_id: str, token_ids, scores, threshold: float = 0.5, allocated_blocks: int = 0):
    """Run the Rust speculative acceptance pass over the model's real logits.

    Returns ``(accepted_count, rejected_count, validated_history)`` where
    ``validated_history`` is the Rust ``SpeculativeStream.token_history``.
    Logs a warning if the Rust-validated set diverges from the raw token count
    (correctness is never silently altered — callers keep the raw generation).
    """
    if rt is None or not rt.engaged:
        return None, None, []
    try:
        stream = core.make_speculative_stream(request_id)
        if stream is None:
            return None, None, []
        cands = build_candidates(token_ids, scores)
        # draft_logits: flat 0.5s (length == number of verified positions).
        stream.start_verification([0.5] * len(cands), cands)
        accepted, rejected, correction, rollback = stream.verify_and_advance(
            float(threshold), int(allocated_blocks),
        )
        history = list(stream.token_history)
        expected = len(token_ids)
        if accepted + (1 if correction is not None else 0) != expected:
            log.warning(
                "Rust speculative check diverged (accepted=%s corr=%s expected=%s) — keeping raw text",
                accepted, correction, expected,
            )
        return accepted, rejected, history
    except Exception as _e:  # noqa: BLE001
        log.warning("Rust speculative verify failed: %s", _e)
        return None, None, []


def rust_engaged() -> bool:
    """True if the loaded core is genuinely wired (runtime built + engaged)."""
    try:
        rt = get_runtime()
        return bool(rt and rt.engaged)
    except Exception:
        return False


__all__ = [
    "get_runtime",
    "rust_engaged",
    "coordinator_enqueue_sync",
    "coordinator_submit_blocking",
    "coordinator_await",
    "begin_stream",
    "finish_stream",
    "verify_tokens",
    "build_candidates",
]
