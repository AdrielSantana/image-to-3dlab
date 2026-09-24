"""Tests for the SF3D CPU-baker fallback.

The bug it fixes only shows on an NVIDIA machine whose CUDA toolkit does not match
PyTorch, so a run cannot catch it here. What can be checked: the fallback moves tensors
to the CPU only when the kernel refuses them, hands the result back on the original
device, and the patched upstream file really routes both ops through it.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_sf3d_cpu_baker.py"


def _load():
    spec = importlib.util.spec_from_file_location("patch_sf3d_cpu_baker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = _load()

# Upstream's texture_baker/baker.py, trimmed to what the patch touches.
ORIGINAL = '''import torch
import torch.nn as nn
from torch import Tensor


class TextureBaker(nn.Module):
    def __init__(self):
        super().__init__()

    def rasterize(
        self,
        uv: Tensor,
        face_indices: Tensor,
        bake_resolution: int,
    ) -> Tensor:
        return torch.ops.texture_baker_cpp.rasterize(
            uv, face_indices.to(torch.int32), bake_resolution
        )

    def interpolate(
        self,
        attr: Tensor,
        rast: Tensor,
        face_indices: Tensor,
    ) -> Tensor:
        return torch.ops.texture_baker_cpp.interpolate(
            attr, face_indices.to(torch.int32), rast
        )
'''


class FakeTensor:
    """Just enough of a tensor: a device, and moves between devices."""

    def __init__(self, device: str, tag: str = ""):
        self.device, self.tag = device, tag

    def cpu(self):
        return FakeTensor("cpu", self.tag)

    def to(self, device):
        return device if isinstance(device, FakeTensor) else FakeTensor(device, self.tag)


def cpu_only_kernel(calls: list):
    """A baker op built with only its CPU kernel, as the mismatch route builds it."""
    def op(*args):
        calls.append([getattr(a, "device", None) for a in args])
        if any(getattr(a, "device", "cpu") != "cpu" for a in args):
            raise NotImplementedError("Could not run 'texture_baker_cpp::rasterize' "
                                      "with arguments from the 'CUDA' backend")
        return FakeTensor("cpu", "result")
    return op


def test_gpu_inputs_are_baked_on_the_cpu_and_returned_to_the_gpu():
    calls = []
    out = patch.run_where_the_kernel_is(cpu_only_kernel(calls), FakeTensor("cuda"), 1024)
    assert out.device == "cuda" and out.tag == "result"
    assert calls == [["cuda", None], ["cpu", None]]


def test_a_kernel_that_accepts_the_inputs_is_called_once():
    calls = []
    out = patch.run_where_the_kernel_is(cpu_only_kernel(calls), FakeTensor("cpu"), 1024)
    assert out.device == "cpu"
    assert len(calls) == 1


def test_other_errors_are_not_swallowed():
    def broken(*args):
        raise RuntimeError("out of memory")
    with pytest.raises(RuntimeError):
        patch.run_where_the_kernel_is(broken, FakeTensor("cuda"))


def test_patched_baker_routes_both_ops_through_the_fallback(tmp_path):
    pytest.importorskip("torch")  # the patched file imports it
    target = tmp_path / "baker.py"
    target.write_text(patch.apply(ORIGINAL))
    spec = importlib.util.spec_from_file_location("patched_baker", target)
    baker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baker)  # the patched file must at least import

    calls = []
    fake_ops = types.SimpleNamespace(rasterize=cpu_only_kernel(calls),
                                     interpolate=cpu_only_kernel(calls))
    baker.torch = types.SimpleNamespace(ops=types.SimpleNamespace(
        texture_baker_cpp=fake_ops), int32="int32")
    faces = FakeTensor("cuda")
    faces.to = lambda dtype: FakeTensor("cuda")  # .to(torch.int32) keeps the device

    rast = baker.TextureBaker().rasterize(FakeTensor("cuda"), faces, 64)
    out = baker.TextureBaker().interpolate(FakeTensor("cuda"), rast, faces)
    assert rast.device == "cuda" and out.device == "cuda"
    assert len(calls) == 4  # each op: refused on the GPU, then run on the CPU


def test_apply_is_idempotent():
    once = patch.apply(ORIGINAL)
    assert patch.is_patched(once)
    assert patch.apply(once) == once
    assert once.count("def run_where_the_kernel_is") == 1


def test_a_moved_anchor_fails_loudly():
    with pytest.raises(SystemExit, match="anchor not found"):
        patch.apply(ORIGINAL.replace("texture_baker_cpp.interpolate(", "baker_v2.interp("))

