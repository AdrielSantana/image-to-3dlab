"""Tests for the importable MLX attention helpers.

The packing maths is tested unconditionally because it is pure. The numerical agreement
test needs both torch and mlx present and is skipped otherwise, so the suite still runs in
an environment that has neither.
"""

from __future__ import annotations

import pytest

from image_to_3dlab import mlx_attention


def _torch():
    import torch

    return torch


def test_split_offsets_produces_contiguous_non_overlapping_slices():
    assert mlx_attention.split_offsets([3, 2, 4]) == [(0, 3), (3, 5), (5, 9)]


def test_split_offsets_handles_empty_and_single():
    assert mlx_attention.split_offsets([]) == []
    assert mlx_attention.split_offsets([7]) == [(0, 7)]


def test_split_offsets_covers_every_token_exactly_once():
    lengths = [5, 1, 9, 2]
    offsets = mlx_attention.split_offsets(lengths)
    covered = [i for start, stop in offsets for i in range(start, stop)]
    assert covered == list(range(sum(lengths)))


def test_memory_snapshot_is_empty_without_mlx(monkeypatch):
    """Callers must not need a guard: no MLX means nothing is held, so nothing to report."""
    monkeypatch.setattr(mlx_attention, "mlx_available", lambda: False)
    assert mlx_attention.memory_snapshot() == {}


def test_release_memory_is_a_noop_without_mlx(monkeypatch):
    monkeypatch.setattr(mlx_attention, "mlx_available", lambda: False)
    assert mlx_attention.release_memory() == {}


def test_release_memory_reports_what_it_freed(monkeypatch):
    """It reports rather than acting silently.

    The size of the cache at the sampling/decode boundary is the measurement that decides
    whether MLX residency explains the decode failures, so a silent call answers nothing.
    """
    snapshots = iter([
        {"active": 1_000, "cache": 5_000_000_000, "peak": 7_000_000_000},
        {"active": 1_000, "cache": 0, "peak": 7_000_000_000},
    ])
    monkeypatch.setattr(mlx_attention, "mlx_available", lambda: True)
    monkeypatch.setattr(mlx_attention, "memory_snapshot", lambda: next(snapshots))
    # `import mlx.core as mx` resolves the parent package first, so both entries are needed.
    import sys as _sys
    import types as _types

    fake_core = _types.ModuleType("mlx.core")
    fake_core.clear_cache = lambda: None
    fake_pkg = _types.ModuleType("mlx")
    fake_pkg.core = fake_core
    monkeypatch.setitem(_sys.modules, "mlx", fake_pkg)
    monkeypatch.setitem(_sys.modules, "mlx.core", fake_core)
    result = mlx_attention.release_memory()
    assert result["released"] == 5_000_000_000
    assert result["cache_after"] == 0
    assert result["peak"] == 7_000_000_000


# Gated per test, not at module level: a module-level skip would silently disable the pure
# tests above in any environment without torch or mlx -- which is most of them, including
# the interpreter the suite normally runs under.
def _have(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


needs_backends = pytest.mark.skipif(
    not (_have("torch") and mlx_attention.mlx_available()),
    reason="numerical checks need both torch and mlx in this interpreter",
)


@needs_backends
def test_resolve_dtype_defaults_to_fp32(monkeypatch):
    import mlx.core as mx

    monkeypatch.delenv(mlx_attention.DTYPE_ENV, raising=False)
    # fp32 by default is a deliberate choice: the first integration must not change
    # numerics, only speed.
    assert mlx_attention.resolve_dtype() is mx.float32


@needs_backends
def test_resolve_dtype_reads_the_environment(monkeypatch):
    import mlx.core as mx

    monkeypatch.setenv(mlx_attention.DTYPE_ENV, "fp16")
    assert mlx_attention.resolve_dtype() is mx.float16


@needs_backends
def test_resolve_dtype_rejects_nonsense(monkeypatch):
    monkeypatch.setenv(mlx_attention.DTYPE_ENV, "float8_banana")
    with pytest.raises(ValueError, match="unsupported"):
        mlx_attention.resolve_dtype()


def _reference(q, k, v, q_seqlen, kv_seqlen):
    """torch SDPA over the same block-diagonal layout, as the ground truth."""
    import torch
    import torch.nn.functional as F

    parts = []
    q_cursor = kv_cursor = 0
    for q_len, kv_len in zip(q_seqlen, kv_seqlen):
        qq = q[q_cursor:q_cursor + q_len].permute(1, 0, 2).unsqueeze(0)
        kk = k[kv_cursor:kv_cursor + kv_len].permute(1, 0, 2).unsqueeze(0)
        vv = v[kv_cursor:kv_cursor + kv_len].permute(1, 0, 2).unsqueeze(0)
        parts.append(F.scaled_dot_product_attention(qq, kk, vv).squeeze(0).permute(1, 0, 2))
        q_cursor += q_len
        kv_cursor += kv_len
    return torch.cat(parts, dim=0)


@needs_backends
def test_varlen_attention_matches_torch_sdpa():
    torch = _torch()
    torch.manual_seed(0)
    heads, dim = 4, 128
    q_seqlen = kv_seqlen = [12, 7]
    total = sum(q_seqlen)
    q, k, v = (torch.randn(total, heads, dim) for _ in range(3))

    got = mlx_attention.varlen_attention(q, k, v, q_seqlen, kv_seqlen)
    want = _reference(q, k, v, q_seqlen, kv_seqlen)

    assert got.shape == want.shape
    # Tolerance is set from measurement, not hope. MLX and torch diverge at fp32 by
    # ~1e-3 on standard-normal inputs and ~1e-4 on half-scale inputs; the gap tracks
    # input magnitude (softmax sharpness), not sequence length. It is NOT the fused
    # kernel: MLX's own matmul-plus-softmax shows the same divergence, so this is
    # MLX-vs-torch fp32 arithmetic on Metal. Documented in
    # journal/lane-a-hypothesis-2026-09-20.md.
    assert torch.allclose(got, want, atol=5e-3), (got - want).abs().max().item()


@needs_backends
def test_varlen_attention_respects_sequence_boundaries():
    """A token in one packed sequence must not see keys from another.

    If the block-diagonal split were wrong this would still produce plausible-looking
    numbers, which is exactly why it is asserted rather than eyeballed.
    """
    torch = _torch()
    torch.manual_seed(1)
    heads, dim = 2, 128
    q = torch.randn(6, heads, dim)
    k = torch.randn(6, heads, dim)
    v = torch.randn(6, heads, dim)

    split = mlx_attention.varlen_attention(q, k, v, [3, 3], [3, 3])
    first_alone = mlx_attention.varlen_attention(q[:3], k[:3], v[:3], [3], [3])

    assert torch.allclose(split[:3], first_alone, atol=5e-3)


@needs_backends
def test_varlen_attention_rejects_mismatched_sequence_counts():
    torch = _torch()
    q = torch.randn(4, 2, 128)
    with pytest.raises(ValueError, match="same number of sequences"):
        mlx_attention.varlen_attention(q, q, q, [2, 2], [4])


@needs_backends
def test_cross_attention_shape_differs_between_q_and_kv():
    """Cross-attention packs different q and kv lengths; the output follows q."""
    torch = _torch()
    torch.manual_seed(2)
    heads, dim = 2, 128
    q = torch.randn(5, heads, dim)
    k = torch.randn(9, heads, dim)
    v = torch.randn(9, heads, dim)
    out = mlx_attention.varlen_attention(q, k, v, [5], [9])
    assert out.shape == (5, heads, dim)
