#!/usr/bin/env python3
"""Finish every prop from a prop sheet: re-baked LODs with normal maps, then compressed.

    python scripts/finish_props.py SPLIT_DIR OUT_DIR [--lods 5000,2500,1000]

Takes the per-prop GLBs `scripts/blender_split_props.py` wrote and gives each prop levels
of detail (LODs): lighter copies a game swaps in as the prop gets further away. The
walkthrough is in `docs/prop-sheets.md`.

**Every LOD is baked from the original, not simplified from LOD0.** Measured on the
barrel of the test sheet: meshoptimizer's simplifier took LOD0 from 5,000 triangles to
1,000 by collapsing edges across texture seams, and the iron hoops came out blotched. A
normal map re-baked for that mesh did not help, because the texture coordinates
themselves had been stretched. Running `blender_retopo_bake.py` again at 1,000 faces,
from the original, with its own atlas and normal map, came out clean on all nine props.
The cost is one atlas per LOD rather than one shared between them.

**Then gltfpack, for size.** meshoptimizer's `gltfpack` reorders each mesh for the GPU,
compresses it and re-encodes its textures as WebP, which is where the weight is: the
chest's LOD0 went from 2.1 MB to 272 KB. Those files need `EXT_meshopt_compression` and
`EXT_texture_webp`, which three.js reads; check your engine before shipping them. The
uncompressed GLBs are always kept beside them.

gltfpack is optional and never downloaded by this script: put it on PATH or pass
`--gltfpack`. Use a native release build (github.com/zeux/meshoptimizer/releases); the
npm build cannot write WebP.

Writes `OUT_DIR/<prop>/<prop>_LOD<n>.glb`, the compressed `<prop>_LOD<n>.web.glb` when
gltfpack is found, and a record of the run in `OUT_DIR/finish_props.json`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retopo_repaint import BLENDER, _run, retopo_command, reuse

REPO = Path(__file__).resolve().parents[1]

# 5,000 triangles kept the chest's rivets and lock readable with a normal map; 2,500 and
# 1,000 are the middle and far steps. `blender_retopo_bake.py` accepts 1,000 to 200,000.
DEFAULT_LODS = (5000, 2500, 1000)
FACE_RANGE = (1000, 200000)
# 1024 is a common prop texture size, and what every measurement here used.
DEFAULT_ATLAS = 1024
ATLAS_SIZES = (1024, 2048, 4096)
# `blender_retopo_bake.py`'s own defaults for the unwrap angle and voxel size.
ANGLE = 89.0
VOXEL = 0.004
GLTFPACK_FLAGS = ("-cc", "-tw")


def parse_lods(text: str) -> list[int]:
    """``"5000,2500,1000"`` to ``[5000, 2500, 1000]``: in range, and each smaller than the last."""
    try:
        lods = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise SystemExit(f"--lods wants comma-separated face counts, got {text!r}") from None
    if not lods:
        raise SystemExit("--lods needs at least one face count")
    low, high = FACE_RANGE
    for faces in lods:
        if not low <= faces <= high:
            raise SystemExit(f"each LOD must be {low:,}..{high:,} faces, got {faces:,}")
    if any(later >= earlier for earlier, later in pairwise(lods)):
        raise SystemExit(f"LODs go from most detailed to least, got {lods}")
    return lods


def collect_props(source: Path) -> list[Path]:
    """The GLBs to finish: every ``*.glb`` in a directory, sorted, or the one file given."""
    if source.is_dir():
        found = sorted(source.glob("*.glb"))
    elif source.suffix.lower() == ".glb" and source.is_file():
        found = [source]
    else:
        raise SystemExit(f"not a GLB or a directory: {source}")
    if not found:
        raise SystemExit(f"no GLBs to finish in {source}")
    return found


def lod_path(out_dir: Path, name: str, index: int, web: bool = False) -> Path:
    suffix = ".web.glb" if web else ".glb"
    return out_dir / name / f"{name}_LOD{index}{suffix}"


def find_gltfpack(explicit: Path | None = None, which=shutil.which) -> Path | None:
    """``--gltfpack`` if given, else ``gltfpack`` on PATH, else ``vendor/gltfpack/``."""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"--gltfpack {explicit} does not exist")
        return explicit
    on_path = which("gltfpack")
    if on_path:
        return Path(on_path)
    vendored = REPO / "vendor" / "gltfpack" / "gltfpack"
    return vendored if vendored.is_file() else None


def gltfpack_command(binary: Path, source: Path, output: Path) -> list[str]:
    """Compress mesh and textures; no simplification, the LODs are already baked."""
    return [str(binary), "-i", str(source), "-o", str(output), *GLTFPACK_FLAGS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path,
                        help="the directory blender_split_props.py wrote, or one prop GLB")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--lods", default=",".join(map(str, DEFAULT_LODS)),
                        help="faces per LOD, most detailed first")
    parser.add_argument("--atlas", type=int, default=DEFAULT_ATLAS, choices=ATLAS_SIZES)
    parser.add_argument("--metallic", type=float, default=0.25)
    parser.add_argument("--roughness", type=float, default=0.65)
    parser.add_argument("--ior", type=float, default=1.45)
    parser.add_argument("--gltfpack", type=Path, default=None)
    parser.add_argument("--no-compress", action="store_true",
                        help="skip gltfpack even when it is available")
    parser.add_argument("--resume", action="store_true",
                        help="keep LODs already written instead of baking them again")
    parser.add_argument("--blender", type=Path, default=BLENDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir
    props = collect_props(args.source)
    lods = parse_lods(args.lods)
    gltfpack = None if args.no_compress else find_gltfpack(args.gltfpack)
    if gltfpack is None and not args.no_compress:
        print("gltfpack not found: writing uncompressed LODs only (see --help)")

    record = {"lods": lods, "atlas": args.atlas, "gltfpack": str(gltfpack) if gltfpack else None,
              "surface": {"metallic": args.metallic, "roughness": args.roughness,
                          "ior": args.ior},
              "props": []}
    started = time.time()
    for source in props:
        name = source.stem
        logs = out_dir / name / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        entry = {"name": name, "source": str(source), "lods": []}
        for index, faces in enumerate(lods):
            step = time.time()
            baked = lod_path(out_dir, name, index)
            if not reuse(baked, args.resume):
                _run(retopo_command(source, baked, faces, args.atlas, ANGLE, VOXEL,
                                    args.metallic, args.roughness, args.ior,
                                    blender=args.blender, normal_map=True),
                     logs / f"LOD{index}.log", f"{name} LOD{index}")
            lod = {"faces": faces, "glb": str(baked), "bytes": baked.stat().st_size}
            if gltfpack is not None:
                packed = lod_path(out_dir, name, index, web=True)
                _run(gltfpack_command(gltfpack, baked, packed),
                     logs / f"LOD{index}.gltfpack.log", f"{name} LOD{index} gltfpack")
                lod.update(web_glb=str(packed), web_bytes=packed.stat().st_size)
            lod["seconds"] = round(time.time() - step, 1)
            entry["lods"].append(lod)
        record["props"].append(entry)
        sizes = ", ".join(f"LOD{i} {lod.get('web_bytes', lod['bytes']) / 1024:,.0f} KB"
                          for i, lod in enumerate(entry["lods"]))
        print(f"{name}: {sizes}", flush=True)

    record["total_seconds"] = round(time.time() - started, 1)
    (out_dir / "finish_props.json").write_text(json.dumps(record, indent=2))
    print(f"finished {len(props)} props in {record['total_seconds']:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
