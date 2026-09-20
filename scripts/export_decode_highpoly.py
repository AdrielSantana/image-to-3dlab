#!/usr/bin/env python3
"""Export a cached decode as a high-poly PLY, to bake detail from.

    .venv/bin/python scripts/export_decode_highpoly.py decode.pt high.ply

**Why this exists.** The decode holds the mesh as the model actually produced it — 19.2
million faces on the Snag — and `generate.py` throws ~98% of that away before it ever
reaches a GLB. A normal-map bake needs exactly that discarded detail, and it is already on
disk: no sculpting step, no second generation. This turns the cached `.pt` into the PLY
that `blender_bake_normals.py` takes as its source.

Two things happen on the way out, and one deliberately does not:

* **Merge by position.** The decoder splits vertices along attribute seams, so the surface
  arrives in pieces that cannot see their neighbours. Nothing that walks across the
  surface — connectivity, winding repair — works until they are welded.
* **Repair winding.** A bake reads the source's *normals*. The decoder's winding is
  inconsistent and in many components, so patches of the map would come out inverted.
  `repair_decode.repair` orients each component by its own signed volume. On the Snag this
  was not marginal: 100,586 of 215,842 components were inside-out.
* **No decimation by default.** `--target-faces` exists and is off, because measuring it
  settled the question: the welded Snag decode is 19,172,397 faces in **1,072** connected
  components, 99.7% of them in one. Collapsing it to 1.5M with fast_simplification
  shattered that into **215,842** components — 20% of the result in fragments under ten
  faces each. Baking against that produces rainbow confetti, not detail: rays leave the
  low-poly, hit a fragment floating just off the surface, and record its arbitrary normal.
  A 2048px map cannot resolve 19M faces either, but too much source is free and a
  shattered source is not. If you do decimate, the component counts are reported, so the
  damage is visible rather than silent.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _repair_module():
    """Load `repair_decode` by path, so this runs from any working directory."""
    spec = importlib.util.spec_from_file_location(
        "repair_decode", SCRIPTS / "repair_decode.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def component_count(vertices, faces) -> int:
    """How many disconnected pieces the surface is in.

    Counted over the *vertex* graph rather than face adjacency: on a 19M-face mesh that
    is the difference between seconds and minutes, and the answer is the same. This is
    the number that exposes a simplifier shattering a mesh it was asked to reduce.
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return 0
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])
    )
    graph = sp.coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    )
    _count, labels = connected_components(graph, directed=False)
    # Isolated vertices are their own components and no part of the surface.
    return len(np.unique(labels[faces[:, 0]]))


def decimate(vertices, faces, target_faces: int):
    """Collapse to `target_faces`, or pass the mesh through unchanged if already smaller.

    Returns (vertices, faces). Kept separate and off by default: see the module docstring
    for what it did to the Snag.
    """
    import fast_simplification
    import numpy as np

    if not target_faces or len(faces) <= target_faces:
        return vertices, faces
    out_vertices, out_faces = fast_simplification.simplify(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        target_count=int(target_faces),
    )
    return out_vertices, out_faces


def prepare_highpoly(vertices, faces, target_faces: int = 0, repair: bool = True):
    """Weld, optionally decimate, then orient — a mesh a normal bake can be trusted against.

    Returns (mesh, stats). `stats` records the face count at each step, the component
    count either side of any decimation, and how many components were inside-out before
    the repair — the numbers that say whether the bake source is safe to use.
    """
    import numpy as np
    import trimesh

    stats: dict[str, object] = {"faces_in": len(faces)}

    welded = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    stats["faces_welded"] = len(welded.faces)
    stats["components_welded"] = component_count(welded.vertices, welded.faces)

    out_vertices, out_faces = decimate(welded.vertices, welded.faces, target_faces)
    if len(out_faces) != len(welded.faces):
        stats["faces_decimated"] = len(out_faces)
        stats["components_decimated"] = component_count(out_vertices, out_faces)

    if repair:
        repair_decode = _repair_module()
        mesh, before, after = repair_decode.repair(
            np.ascontiguousarray(out_vertices), np.ascontiguousarray(out_faces)
        )
        stats["inverted_components_before"] = before["inverted_components"]
        stats["inverted_components_after"] = after["inverted_components"]
    else:
        mesh = trimesh.Trimesh(
            vertices=np.ascontiguousarray(out_vertices),
            faces=np.ascontiguousarray(out_faces),
            process=False,
        )

    stats["faces_out"] = len(mesh.faces)
    stats["extents"] = [round(float(e), 5) for e in mesh.extents]
    return mesh, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("decode", type=Path, help="cached decode .pt")
    parser.add_argument("output", type=Path, help="PLY to write")
    parser.add_argument(
        "--target-faces", type=int, default=0,
        help="Decimate to this many faces first. Off by default: fast_simplification "
             "shatters these meshes, and a bake against fragments is confetti. Watch the "
             "reported component count if you turn it on",
    )
    parser.add_argument(
        "--no-repair", action="store_true",
        help="Skip the per-component winding repair. Faster, and wrong wherever the "
             "decoder left a component inside-out",
    )
    args = parser.parse_args()

    import torch

    started = time.time()
    payload = torch.load(args.decode, map_location="cpu", weights_only=False)
    print(f"loaded {payload['faces'].shape[0]:,} faces", flush=True)

    mesh, stats = prepare_highpoly(
        payload["vertices"].cpu().numpy(),
        payload["faces"].cpu().numpy(),
        target_faces=args.target_faces,
        repair=not args.no_repair,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    print(f"wrote {args.output} in {time.time() - started:.0f}s", flush=True)
    for key, value in stats.items():
        print(f"  {key}: {value}", flush=True)
    if "components_decimated" in stats:
        shattered = stats["components_decimated"] / max(stats["components_welded"], 1)
        if shattered > 2:
            print(
                f"WARNING: decimation multiplied the component count by {shattered:.0f}x. "
                "The source is now fragments; a bake against it will be noise.",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
