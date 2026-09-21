#!/usr/bin/env python3
"""End-to-end Pixal3D generation on Apple Silicon: image -> textured GLB.

    python scripts/pixal3d_generate.py input.png output.glb [--res 1024] [--seed 42]

Wraps `trellis-cli` from `vendor/pixal3d-cpp` (raven38/pixal3d.cpp), a C++/GGML runtime
with Metal kernels. Build it with `scripts/bootstrap_pixal3d_cpp.sh`.

**Why this port and not the PyTorch one.** `pawel-mazurkiewicz/Pixal3D-mac` loads ~22 GB of
bf16 weights before sampling and its low-VRAM mode moves models between CPU and GPU, which
frees nothing on unified memory. This one runs the same model from 8 GB of Q8_0 weights,
with real Metal flash-attention, and produced the moss fox in 5m50s where the PyTorch port
could not finish on a 32 GB machine. See `docs/pixal3d-evaluation-2026-09-20.md`.

**Single-view needs a camera.** Pixal3D conditions on pixel-aligned features projected
through an explicit camera, so `--sv-image` synthesizes a front gauge camera at `--fov`
(20 degrees by default). A pre-matted RGBA image goes straight in; anything else is matted
first with BiRefNet, which costs ~13s and changes the cutout, so alpha is preferred.

Deliberately not `--pipeline-type 512`: the single-view weight family has no res-512
texture flow.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = REPO / "vendor" / "pixal3d-cpp"
CLI = PIXAL3D_ROOT / "build" / "trellis-cli"
MODELS = PIXAL3D_ROOT / "models" / "pixal3d-sv"

# The gauge camera the single-view path is designed around: 20 degrees, as radians.
DEFAULT_FOV = 0.3490658503988659

# Structure guidance strength. `trellis-cli` defaults to 7.5; at that setting the warrior
# girl lost her sword blade entirely and 10 brought it back with tighter proportions, so
# 10 is the default here. 13 is worse -- the blade detaches from the hand.
# See docs/pixal3d-evaluation-2026-09-20.md.
DEFAULT_GSS = 10.0

STAGES = ["stage", "views", "ss", "shape", "decode", "texture", "write"]
STAGE_LABELS = {
    "views": "Preparing view",
    "ss": "Sparse structure",
    "shape": "Shape SLAT (512 -> 1024 cascade)",
    "decode": "Shape decode",
    "texture": "Texture SLAT + PBR decode",
    "write": "Writing GLB",
}
# `trellis-cli` announces progress as `[n/6] ...`; this maps n to a stage id.
BANNER_STAGES = {1: "views", 2: "ss", 3: "shape", 4: "decode", 5: "texture", 6: "write"}


def has_alpha(image: Path) -> bool:
    """Whether the image carries a matte already.

    A pre-matted RGBA image skips BiRefNet entirely, which is both faster and a better
    comparison: the cutout is then identical to whatever else was run on that image.
    """
    from PIL import Image

    with Image.open(image) as opened:
        return opened.mode in ("RGBA", "LA") or "transparency" in opened.info


def build_command(
    image: Path, output: Path, res: int, seed: int, fov: float,
    models: Path = MODELS, cli: Path = CLI, matted: bool = True,
    gss: float = DEFAULT_GSS, gsh: float | None = None,
) -> list[str]:
    """The `trellis-cli` invocation.

    Pre-matted images take `--sv-image`, which crops to the alpha bounding box the way the
    reference preprocess does and synthesizes the gauge camera. Everything else goes in
    positionally and is matted by BiRefNet first.

    `--gss` is always passed rather than left to the CLI default, because that default
    (7.5) is the setting that dropped the warrior girl's sword blade.
    """
    if matted:
        head = [str(cli), "--sv-image", str(image)]
    else:
        head = [str(cli), str(image), "--bg-removal", "birefnet"]
    command = head + [
        "--fov", str(fov),
        "--models", str(models),
        "--seed", str(seed),
        "--res", str(res),
        "--pixal3d-weights", "sv",
        "--gss", str(gss),
    ]
    if gsh is not None:
        command += ["--gsh", str(gsh)]
    return command + [str(output)]


def stage_from_banner(line: str) -> tuple[str, int] | None:
    """Turn a `[n/6] ...` banner into (stage_id, percent), or None.

    Percent is the banner index rather than anything measured: the stages are wildly
    uneven (shape is ~2 minutes, decode ~13 seconds), but a monotonic bar beats none.
    """
    if not line.startswith("[") or "]" not in line:
        return None
    marker = line[1:line.index("]")]
    if "/6" not in marker:
        return None
    try:
        index = int(marker.split("/")[0])
    except ValueError:
        return None
    stage = BANNER_STAGES.get(index)
    if stage is None:
        return None
    return stage, min(99, round(index / 6 * 100))


def readiness(cli: Path = CLI, models: Path = MODELS) -> dict[str, object]:
    """What is missing before a run can start, for the viewer's setup panel."""
    weights = sorted(models.glob("*.gguf")) if models.is_dir() else []
    return {
        "cli_built": cli.is_file(),
        "cli_path": str(cli),
        "models_dir": str(models),
        "weights_present": len(weights),
        "ready": cli.is_file() and len(weights) >= 9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--res", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fov", type=float, default=DEFAULT_FOV,
        help="gauge camera FOV in radians for the single-view rig; 0.349 is 20 degrees",
    )
    parser.add_argument(
        "--gss", type=float, default=DEFAULT_GSS,
        help="structure guidance strength; 10 recovers thin props the CLI default of 7.5 drops",
    )
    parser.add_argument(
        "--gsh", type=float, default=None,
        help="shape guidance strength; left to the runtime default when unset",
    )
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--cli", type=Path, default=CLI)
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"not found: {args.image}")
    state = readiness(args.cli, args.models)
    if not state["ready"]:
        raise SystemExit(
            f"pixal3d.cpp is not ready: {state}. Run scripts/bootstrap_pixal3d_cpp.sh"
        )

    matted = has_alpha(args.image)
    if not matted:
        print("[pixal3d] no alpha channel; BiRefNet will matte it first (~13s)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # `trellis-cli` is launched from its own tree so it can find its Metal library, which
    # means a relative input or output path would resolve against *that* directory and the
    # run dies at once with "can't fopen". Absolute paths are the only safe thing to pass.
    args.image = args.image.resolve()
    args.output = args.output.resolve()

    started = time.time()
    command = build_command(
        args.image, args.output, args.res, args.seed, args.fov,
        args.models, args.cli, matted, args.gss, args.gsh,
    )
    print(f"[pixal3d] res={args.res} seed={args.seed} gss={args.gss} matted={matted}",
          flush=True)

    process = subprocess.Popen(
        command, cwd=str(PIXAL3D_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip("\n")
        # ggml logs every Metal pipeline it compiles; that is hundreds of lines of noise.
        if line.startswith("ggml_metal") or "loaded kernel" in line:
            continue
        print(line, flush=True)
    code = process.wait()
    if code != 0:
        raise SystemExit(f"trellis-cli exited with code {code}")
    if not args.output.is_file():
        raise SystemExit(f"trellis-cli exited 0 without writing {args.output}")

    size = args.output.stat().st_size / 1048576
    print(f"[pixal3d] done in {time.time() - started:.0f}s -> {args.output} "
          f"({size:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
