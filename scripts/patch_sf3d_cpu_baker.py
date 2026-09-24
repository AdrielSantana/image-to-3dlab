#!/usr/bin/env python3
"""Let SF3D's texture baker run on the CPU while the model runs on an NVIDIA GPU.

    python scripts/patch_sf3d_cpu_baker.py            # applies to vendor/stable-fast-3d
    python scripts/patch_sf3d_cpu_baker.py --check    # exit 1 if not patched

**Why.** On Linux the baker's CUDA kernel only builds when the CUDA toolkit matches the
CUDA that PyTorch was built for. When it does not (a fresh install's PyTorch is CUDA 13,
many machines have a 12.x toolkit), `bootstrap_sf3d.py` builds the baker's CPU kernel
instead. But SF3D hands the baker tensors that live on the GPU, and a CPU-only op refuses
them: "Could not run 'texture_baker_cpp::rasterize' with arguments from the 'CUDA'
backend". Found on a 3090 Ti pod, 2026-09-24.

**What it does.** Each baker op is tried as is. Only if PyTorch says the op has no kernel
for that device are the inputs copied to the CPU, and the result is copied back to where
they came from. On a Mac (Metal) or a matching CUDA build the first try succeeds and
nothing changes.

`vendor/` is git-ignored, so `bootstrap_sf3d.py` applies this after cloning and before
building. Safe to run repeatedly: it detects its own marker and does nothing twice.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

MARKER = "i2l_cpu_baker"

DEFAULT_TARGET = Path("vendor/stable-fast-3d/texture_baker/texture_baker/baker.py")

IMPORT_ANCHOR = "from torch import Tensor\n"
OPS = ("rasterize", "interpolate")


def run_where_the_kernel_is(op, *args):
    """Run a baker op, falling back to its CPU kernel when it has none for the inputs'
    device, and hand the result back on that device."""
    try:
        return op(*args)
    except NotImplementedError:
        device = next(a.device for a in args if hasattr(a, "cpu"))
        result = op(*(a.cpu() if hasattr(a, "cpu") else a for a in args))
        return result.to(device)


def _call(op: str) -> str:
    return f"torch.ops.texture_baker_cpp.{op}(\n"


def is_patched(source: str) -> bool:
    return MARKER in source


def apply(source: str) -> str:
    """Return the patched source. Idempotent."""
    if is_patched(source):
        return source
    missing = [a for a in (IMPORT_ANCHOR, *map(_call, OPS)) if a not in source]
    if missing:
        raise SystemExit(
            f"anchor not found in SF3D's baker.py: {missing[0]!r}. Upstream changed how "
            "the baker calls its kernels; check the CPU-baker route by hand before "
            "trusting it on a machine whose CUDA toolkit does not match PyTorch."
        )
    helper = f"\n\n# {MARKER}: see scripts/patch_sf3d_cpu_baker.py in image-to-3dlab.\n"
    helper += inspect.getsource(run_where_the_kernel_is)
    source = source.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + helper, 1)
    for op in OPS:
        source = source.replace(
            _call(op), f"run_where_the_kernel_is(\n            torch.ops.texture_baker_cpp.{op},\n", 1)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if the patch is not present")
    args = parser.parse_args()

    if not args.target.is_file():
        raise SystemExit(f"not found: {args.target}")
    source = args.target.read_text()

    if args.check:
        print(("patched: " if is_patched(source) else "NOT PATCHED: ") + str(args.target))
        return 0 if is_patched(source) else 1

    patched = apply(source)
    if patched == source:
        print(f"already patched: {args.target}")
        return 0
    args.target.write_text(patched)
    print(f"patched {args.target}: baker ops fall back to the CPU kernel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
