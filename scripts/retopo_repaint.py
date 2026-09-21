#!/usr/bin/env python3
"""Finish a generated asset: retopologise, repaint, compress — one command.

    python scripts/retopo_repaint.py generated.glb source.png finished.glb

**Why this exists.** The route that works on this repo's assets is four separate commands
run by hand, each with its own flags, in an order that matters:

    blender_retopo_bake.py → run_paint_pbr.py → compress_glb_textures.py

Run by hand across several assets, each gets slightly different treatment and the
comparison between them stops meaning anything. It also takes ~8 minutes of attention per
asset, most of it waiting. This runs the chain, keeps every intermediate, and writes a
JSON record of what was actually done.

**What the route is for.** TRELLIS.2's geometry is good and its 1024 texture stage bleaches
flat-illustration inputs. So: take the geometry, throw the texture away, and repaint from
the source image with Hunyuan 2.1 PBR. Retopology in the middle is what makes the repaint
affordable — xatlas takes 12 seconds on a 40k mesh against minutes at 500k, and the paint
stage's face-count ceiling stops being reachable.

**Stages, and why each is where it is:**

1. **Retopologise** (`blender_retopo_bake.py`, headless Blender). Voxel-remesh first, then
   decimate. The ordering is load-bearing: decimating the raw mesh shatters thin geometry,
   measured. Also bakes the original texture across as a fallback, so the intermediate is
   viewable even if the paint stage fails.
2. **Repaint** (`hunyuan_mlx/paint`, its own venv). Reuses `run_paint` from
   `hunyuan_mlx_xiong_generate.py` rather than re-deriving the env-var contract.
3. **Compress** (`compress_glb_textures.py`). The paint stage emits two uncompressed
   4096² PNGs, which is 30 MB of a 32 MB asset.

**Surface response is per-asset and this does not guess it.** Only base colour is baked, so
`--metallic/--roughness/--ior` stand in for the source's material. The defaults are neutral
organic; the Snag wanted 0.648 / 0.686 / 1.4 by eye. Whatever is used is recorded in the
JSON so the next asset can be judged against a known setting rather than a remembered one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")

# The viewer's job API parses these to drive its progress panel. Same contract as
# `scripts/blender_rebind.py`, so one parser serves both.
STAGE_MARKER = "I2L_STAGE"


def emit_stage(phase: str, message: str) -> None:
    """Announce a stage boundary on stdout, for whatever is watching."""
    print(f"{STAGE_MARKER}::{phase}::{message}", flush=True)


def stage_plan(skip_paint: bool, skip_compress: bool) -> list[str]:
    """The stages this run will execute, in order.

    Named separately from the running of them so a caller — the viewer's job API, say —
    can show the sequence before anything starts, and so the order is testable.
    """
    stages = ["retopologise"]
    if not skip_paint:
        stages.append("repaint")
    if not skip_compress:
        stages.append("compress")
    return stages


def retopo_command(
    source: Path, output: Path, faces: int, atlas: int, angle: float,
    voxel: float, metallic: float, roughness: float, ior: float,
    blender: Path = BLENDER,
) -> list[str]:
    """The headless Blender invocation, as a list.

    `blender_retopo_bake.py` takes positional arguments after `--`, in a fixed order, and
    getting that order wrong silently produces a differently-tuned asset rather than an
    error. Built here so it can be asserted in a test.
    """
    return [
        str(blender), "--background", "--python",
        str(SCRIPTS / "blender_retopo_bake.py"), "--",
        str(source), str(output), str(faces), str(atlas), str(angle),
        str(voxel), str(metallic), str(roughness), str(ior),
    ]


def reuse(path: Path, resume: bool) -> bool:
    """Whether a stage can be skipped because its artifact is already there.

    A zero-byte file is a stage that died mid-write, not one that finished; treating it
    as complete would hand the next stage a truncated GLB instead of re-running the stage
    that actually failed.
    """
    return resume and path.is_file() and path.stat().st_size > 0


def _run(command: list[str], log: Path, label: str) -> None:
    """Run one stage, tee its output to a log, and fail loudly."""
    print(f"[{label}] {' '.join(command[:4])} ...", flush=True)
    with log.open("w") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-15:])
        raise SystemExit(f"[{label}] failed (exit {process.returncode}); last lines:\n{tail}")


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface, separately from running it.

    Extracted so a test can assert the flags and their defaults without a Blender run:
    every one of these reaches a stage that costs minutes, and a default that silently
    drifts produces a differently-tuned asset rather than an error.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="generated GLB to finish")
    parser.add_argument("image", type=Path, help="source concept art, for the repaint")
    parser.add_argument("output", type=Path, help="finished GLB to write")
    parser.add_argument("--faces", type=int, default=40000, help="retopology target")
    parser.add_argument("--atlas", type=int, default=2048)
    parser.add_argument("--angle", type=float, default=89.0)
    parser.add_argument(
        "--voxel", type=float, default=0.004,
        help="voxel size as a fraction of the asset; 0 skips the remesh, which shatters "
             "thin geometry on everything tested so far",
    )
    parser.add_argument("--metallic", type=float, default=0.25)
    parser.add_argument("--roughness", type=float, default=0.65)
    parser.add_argument("--ior", type=float, default=1.45)
    parser.add_argument("--paint-seed", type=int, default=0)
    parser.add_argument("--paint-res", type=int, default=512)
    parser.add_argument("--paint-steps", type=int, default=15)
    parser.add_argument("--paint-tex", type=int, default=4096)
    parser.add_argument("--texture-size", type=int, default=2048,
                        help="cap for colour textures in the compress stage")
    parser.add_argument("--skip-paint", action="store_true",
                        help="stop after retopology, keeping the transferred texture")
    parser.add_argument("--skip-compress", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse any stage artifact already sitting beside the output instead of "
             "recomputing it. The repaint is five to six minutes of a six-minute run, so "
             "a run that died in compression must not pay for it twice. Only safe with "
             "the settings the intermediates were made with -- change one and start over",
    )
    parser.add_argument("--blender", type=Path, default=BLENDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    for path in (args.source, args.image):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix("")
    stages = stage_plan(args.skip_paint, args.skip_compress)
    print(f"[retopo-repaint] {args.source.name} -> {args.output.name}: "
          f"{' -> '.join(stages)}", flush=True)

    started = time.time()
    timings: dict[str, float] = {}

    retopo_glb = Path(f"{stem}_retopo.glb")
    step = time.time()
    if reuse(retopo_glb, args.resume):
        emit_stage("retopologise", f"Reusing {retopo_glb.name}")
    else:
        emit_stage("retopologise", f"Retopologising to {args.faces:,} faces")
        _run(
            retopo_command(
                args.source, retopo_glb, args.faces, args.atlas, args.angle,
                args.voxel, args.metallic, args.roughness, args.ior, args.blender,
            ),
            Path(f"{stem}_retopo.log"), "retopologise",
        )
    timings["retopologise"] = round(time.time() - step, 1)

    current = retopo_glb
    if not args.skip_paint:
        painted = Path(f"{stem}_painted.glb")
        step = time.time()
        if reuse(painted, args.resume):
            emit_stage("repaint", f"Reusing {painted.name}")
        else:
            # Imported here, not at the top: it pulls in MLX and the paint venv, which a
            # resume that already has the painted GLB has no reason to wait for.
            sys.path.insert(0, str(SCRIPTS))
            from hunyuan_mlx_xiong_generate import run_paint

            emit_stage("repaint", f"Repainting from {args.image.name} at {args.paint_res}px")
            run_paint(
                current, args.image, painted, args.paint_seed, args.paint_res,
                args.paint_steps, args.paint_tex, started,
            )
        timings["repaint"] = round(time.time() - step, 1)
        current = painted

    if not args.skip_compress:
        sys.path.insert(0, str(SCRIPTS))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "compress_glb_textures", SCRIPTS / "compress_glb_textures.py"
        )
        compress_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compress_module)

        step = time.time()
        emit_stage("compress", f"Compressing textures to {args.texture_size}px")
        data = current.read_bytes()
        out_bytes, report = compress_module.compress(data, max_size=args.texture_size)
        args.output.write_bytes(out_bytes)
        timings["compress"] = round(time.time() - step, 1)
        print(f"[compress] {len(data) / 1048576:.1f} -> "
              f"{len(out_bytes) / 1048576:.1f} MB", flush=True)
    else:
        args.output.write_bytes(current.read_bytes())
        report = []

    record = {
        "resumed": args.resume,
        "source": str(args.source),
        "image": str(args.image),
        "output": str(args.output),
        "stages": stages,
        "seconds": timings,
        "total_seconds": round(time.time() - started, 1),
        "retopology": {"faces": args.faces, "atlas": args.atlas, "voxel": args.voxel,
                       "angle": args.angle},
        "surface": {"metallic": args.metallic, "roughness": args.roughness, "ior": args.ior},
        "paint": None if args.skip_paint else {
            "seed": args.paint_seed, "res": args.paint_res,
            "steps": args.paint_steps, "texture": args.paint_tex,
        },
        "textures": report,
        "size_bytes": args.output.stat().st_size,
    }
    record_path = Path(f"{stem}.retopo-repaint.json")
    record_path.write_text(json.dumps(record, indent=2))

    emit_stage("done", f"Finished at {record['size_bytes'] / 1048576:.1f} MB")
    print(f"[retopo-repaint] done in {record['total_seconds']:.0f}s -> {args.output} "
          f"({record['size_bytes'] / 1048576:.1f} MB); record {record_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
