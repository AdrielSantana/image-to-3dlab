"""Split a multi-prop GLB into one upright, named object per prop, headless.

    blender -b --factory-startup -P scripts/blender_split_props.py -- \\
        IN.glb OUT_DIR [--names barrel crate ...] [--turn chest=90 ...]

**Why.** A *prop sheet* is one generated image holding a grid of separate props. Pixal3D
turns the whole sheet into 3D in a single run: nine props in 14 minutes on an M5 with
16 GB, where a single character took about 18. What comes back is one mesh with every
prop in it, each tipped back and some turned on the spot, so this takes it apart.

The walkthrough and every measurement quoted here are in `docs/prop-sheets.md`.

**The steps, and why each one is there:**

1. **Import with merged vertices.** glTF splits vertices along every UV seam, so
   separating the raw import by loose parts returns thousands of UV islands, not props.
2. **Separate, then regroup.** Loose parts whose front-view boxes touch are one prop: a
   lid that decoded apart from its chest, a splinter beside a stump. On the test sheet,
   27 loose parts became the 9 props asked for.
3. **Drop crumbs** under 1% of the faces. Pixal3D already drops most of them.
4. **Order like reading the image**, top row first and left to right, so `--names` can
   follow the order the prompt listed the props in.
5. **Stand each prop up.** Pixal3D's single-view camera is level, and image models draw
   props from a little above, so Pixal3D tips every prop back to show its top to a level
   camera: 22 to 34 degrees on the test sheet. The tilt with the tightest bounding box
   undoes it. It is a rotation, not a guess at the shape: afterwards the tops and bases
   sat within 3.3 degrees of level, and round props measured as deep as they are wide.
6. **Turn box-like props to face the front.** A crate drawn corner-on comes back rotated
   41 degrees about the vertical, which reads as warped when it is only turned. Only
   props whose footprint shrinks by more than `--yaw-threshold` are turned; a round prop
   has the same footprint at every angle, and turning it would swing its painted front
   away. A box drawn almost exactly corner-on is a tie between front and side, so any
   turn near 45 degrees is flagged in the record, and `--turn NAME=DEGREES` fixes it.
7. **Origin at the bottom centre**, each prop at the origin in its own GLB, and all of
   them lined up along X in the `.blend`.

Pixal3D's single-view output faces +Y once imported (the importer's Y-up to Z-up turn
included), so the viewer's left is +X. That is what the reading order assumes.

Writes `OUT_DIR/props.blend`, one `OUT_DIR/<name>.glb` per prop and `OUT_DIR/props.json`,
a record of what was done to each prop.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# The tilt search. Measured tilts on the test sheets ran 15 to 34 degrees about X and
# under 4 about Y; the windows leave room either side without admitting a quarter turn,
# which would stand a prop on its side.
TILT_X_LIMIT = 45.0
TILT_Y_LIMIT = 20.0
# 6,000 vertices find the same angle as all of them to a tenth of a degree, in a
# fraction of the time. A fixed seed keeps a re-run identical.
TILT_SAMPLE = 6000
# Within +-45 degrees, the smallest turn that squares a box up. Anything beyond is the
# same box a quarter turn round.
YAW_LIMIT = 45.0
# At their best yaw on the test sheets, round props shrank by 1-3% and soft ones (a
# sack, a stump) by up to 9%; box-like ones (crate, chest, anvil, stool) by 16-41%.
YAW_THRESHOLD = 0.10
# Turns this close to 45 degrees are front-or-side ties.
YAW_TIE_MARGIN = 5.0
MIN_SHARE = 0.01


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Arguments after Blender's ``--`` separator. Kept free of ``bpy`` so it is testable."""
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(prog="blender_split_props.py", description=__doc__)
    parser.add_argument("source", type=Path, help="the multi-prop GLB from Pixal3D")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--names", nargs="+", default=[],
                        help="names in reading order: top row first, left to right")
    parser.add_argument("--turn", action="append", default=[], metavar="NAME=DEGREES",
                        help="an extra turn about the vertical, applied last; repeatable")
    parser.add_argument("--no-straighten", dest="straighten", action="store_false")
    parser.add_argument("--yaw-threshold", type=float, default=YAW_THRESHOLD,
                        help="turn a prop only when its footprint shrinks by more than "
                             "this fraction")
    parser.add_argument("--min-share", type=float, default=MIN_SHARE,
                        help="drop parts with less than this fraction of the faces")
    args = parser.parse_args(argv)
    args.turn = parse_turns(args.turn)
    if not 0.0 <= args.yaw_threshold < 1.0:
        parser.error(f"--yaw-threshold is a fraction in [0, 1), got {args.yaw_threshold}")
    if not 0.0 <= args.min_share < 1.0:
        parser.error(f"--min-share is a fraction in [0, 1), got {args.min_share}")
    return args


def parse_turns(items: list[str]) -> dict[str, float]:
    """``["chest=90"]`` to ``{"chest": 90.0}``."""
    turns = {}
    for item in items:
        name, sep, degrees = item.partition("=")
        if not sep or not name:
            raise SystemExit(f"--turn wants NAME=DEGREES, got {item!r}")
        try:
            turns[name] = float(degrees)
        except ValueError:
            raise SystemExit(f"--turn wants a number of degrees, got {item!r}") from None
    return turns


Box = tuple[tuple[float, float, float], tuple[float, float, float]]


def boxes_touch(a: Box, b: Box, margin: float) -> bool:
    """Do two boxes overlap in the front view (X and Z), give or take ``margin``?

    Depth is ignored on purpose: parts of one prop sit at different depths, while two
    props on a sheet never share a front-view cell.
    """
    (alo, ahi), (blo, bhi) = a, b
    return all(alo[i] - margin <= bhi[i] and blo[i] - margin <= ahi[i] for i in (0, 2))


def group_touching(boxes: list[Box], margin: float) -> list[list[int]]:
    """Indices of parts that belong together, found by chaining touching boxes."""
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if find(i) != find(j) and boxes_touch(boxes[i], boxes[j], margin):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def keep_large(face_counts: list[int], share: float = MIN_SHARE) -> list[int]:
    """Indices of parts with at least ``share`` of all the faces."""
    total = sum(face_counts)
    return [i for i, n in enumerate(face_counts) if n >= share * total]


def reading_order(centres: list[tuple[float, float]], heights: list[float]) -> list[int]:
    """Indices ordered like reading the source image: top row first, left to right.

    ``centres`` are (x, z) in Blender space. A new row starts when a prop's centre sits
    more than half the tallest prop below the previous one; within a row, the viewer's
    left is +X.
    """
    if not centres:
        return []
    tolerance = 0.5 * max(heights)
    by_height = sorted(range(len(centres)), key=lambda i: -centres[i][1])
    rows, current = [], [by_height[0]]
    for i in by_height[1:]:
        if abs(centres[i][1] - centres[current[-1]][1]) < tolerance:
            current.append(i)
        else:
            rows.append(current)
            current = [i]
    rows.append(current)
    return [i for row in rows for i in sorted(row, key=lambda i: -centres[i][0])]


def tilt_matrix(x_degrees: float, y_degrees: float) -> np.ndarray:
    """Rotate about X, then about Y."""
    ax, ay = math.radians(x_degrees), math.radians(y_degrees)
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    return ry @ rx


def sample(points: np.ndarray, count: int = TILT_SAMPLE) -> np.ndarray:
    if len(points) <= count:
        return points
    return points[np.random.default_rng(0).choice(len(points), count, replace=False)]


def box_volume(points: np.ndarray) -> float:
    return float(np.prod(points.max(0) - points.min(0)))


def best_tilt(points: np.ndarray, x_limit: float = TILT_X_LIMIT,
              y_limit: float = TILT_Y_LIMIT) -> tuple[float, float]:
    """The (X, Y) tilt in degrees whose rotation gives the tightest bounding box.

    A whole-degree search, then a tenth-of-a-degree one around the best.
    """
    pts = sample(np.asarray(points, dtype=float))

    def volume(ax: float, ay: float) -> float:
        return box_volume(pts @ tilt_matrix(ax, ay).T)

    coarse = min((volume(ax, ay), ax, ay)
                 for ax in np.arange(-x_limit, x_limit + 0.5, 1.0)
                 for ay in np.arange(-y_limit, y_limit + 0.5, 1.0))
    _, cx, cy = coarse
    fine = min((volume(cx + dx, cy + dy), cx + dx, cy + dy)
               for dx in np.arange(-1.0, 1.05, 0.1)
               for dy in np.arange(-1.0, 1.05, 0.1))
    return round(float(fine[1]), 1), round(float(fine[2]), 1)


def rotate_xy(points_xy: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate points counter-clockwise about the vertical axis."""
    t = math.radians(degrees)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return np.asarray(points_xy, dtype=float) @ r.T


def footprint(points_xy: np.ndarray) -> float:
    return float(np.prod(points_xy.max(0) - points_xy.min(0)))


def best_yaw(points_xy: np.ndarray, limit: float = YAW_LIMIT,
             step: float = 0.25) -> tuple[float, float]:
    """The turn about the vertical, in degrees, with the smallest footprint, and how much
    smaller that footprint is than the untouched one (0.41 means 41% smaller)."""
    pts = sample(np.asarray(points_xy, dtype=float))
    base = footprint(pts)
    area, degrees = min((footprint(rotate_xy(pts, d)), float(d))
                        for d in np.arange(-limit, limit, step))
    gain = 1.0 - area / base if base > 0 else 0.0
    return round(degrees, 2), round(gain, 4)


def yaw_is_tie(degrees: float, limit: float = YAW_LIMIT,
               margin: float = YAW_TIE_MARGIN) -> bool:
    """Is this turn so close to 45 degrees that front and side are a coin toss?"""
    return abs(degrees) > limit - margin


def default_name(index: int) -> str:
    return f"prop_{index + 1:02d}"


def main() -> int:
    import bpy
    from mathutils import Matrix, Vector

    args = parse_args(list(sys.argv))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source), merge_vertices=True)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"no mesh found in {args.source}")

    # Bake any importer transform into the vertices, so object space is world space.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    for obj in meshes:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")
    parts = [o for o in bpy.data.objects if o.type == "MESH"]
    print(f"SPLIT:: {len(parts)} loose parts")

    def vertices(obj) -> np.ndarray:
        co = np.empty(len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get("co", co)
        return co.reshape(-1, 3)

    def bounds(obj) -> Box:
        co = vertices(obj)
        return tuple(co.min(0).tolist()), tuple(co.max(0).tolist())

    boxes = [bounds(p) for p in parts]
    extent = max(max(hi[i] for _, hi in boxes) - min(lo[i] for lo, _ in boxes) for i in range(3))
    props = []
    for group in group_touching(boxes, margin=0.01 * extent):
        bpy.ops.object.select_all(action="DESELECT")
        for i in group:
            parts[i].select_set(True)
        bpy.context.view_layer.objects.active = parts[group[0]]
        if len(group) > 1:
            bpy.ops.object.join()
        props.append(bpy.context.view_layer.objects.active)

    keep = set(keep_large([len(p.data.polygons) for p in props], args.min_share))
    for i, prop in enumerate(list(props)):
        if i not in keep:
            bpy.data.objects.remove(prop)
    props = [p for i, p in enumerate(props) if i in keep]

    boxes = [bounds(p) for p in props]
    centres = [((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2) for lo, hi in boxes]
    heights = [hi[2] - lo[2] for lo, hi in boxes]
    props = [props[i] for i in reading_order(centres, heights)]
    if args.names and len(args.names) != len(props):
        print(f"SPLIT:: warning: {len(args.names)} names for {len(props)} props; "
              f"the rest are numbered")

    record, cursor = [], 0.0
    for index, prop in enumerate(props):
        name = args.names[index] if index < len(args.names) else default_name(index)
        prop.name = prop.data.name = name
        entry = {"name": name, "faces": len(prop.data.polygons)}

        if args.straighten:
            tx, ty = best_tilt(vertices(prop))
            prop.data.transform(Matrix([[*row, 0.0] for row in tilt_matrix(tx, ty)] +
                                       [[0.0, 0.0, 0.0, 1.0]]))
            yaw, gain = best_yaw(vertices(prop)[:, :2])
            turned = gain > args.yaw_threshold
            if turned:
                prop.data.transform(Matrix.Rotation(math.radians(yaw), 4, "Z"))
            entry.update(tilt_degrees=[tx, ty], yaw_degrees=yaw if turned else 0.0,
                         yaw_gain=gain, yaw_tie=turned and yaw_is_tie(yaw))
            if entry["yaw_tie"]:
                print(f"SPLIT:: {name}: turned {yaw:+.1f} degrees, close to a front/side "
                      f"tie; check it faces the right way (--turn {name}=90 if not)")
        if name in args.turn:
            prop.data.transform(Matrix.Rotation(math.radians(args.turn[name]), 4, "Z"))
            entry["extra_turn_degrees"] = args.turn[name]

        co = vertices(prop)
        lo, hi = co.min(0), co.max(0)
        prop.data.transform(Matrix.Translation(Vector((-(lo[0] + hi[0]) / 2,
                                                       -(lo[1] + hi[1]) / 2, -lo[2]))))
        prop.data.update()
        size = (hi - lo).tolist()
        entry["size"] = [round(s, 4) for s in size]
        prop.location = (cursor + size[0] / 2, 0.0, 0.0)
        cursor += size[0] + 0.15 * max(heights)
        record.append(entry)
        print(f"SPLIT:: {name}: {entry['faces']:,} faces, "
              f"tilt {entry.get('tilt_degrees')}, turn {entry.get('yaw_degrees', 0.0)}")

    for prop in props:
        home = prop.location.copy()
        prop.location = (0.0, 0.0, 0.0)
        bpy.ops.object.select_all(action="DESELECT")
        prop.select_set(True)
        bpy.context.view_layer.objects.active = prop
        bpy.ops.export_scene.gltf(filepath=str(args.out_dir / f"{prop.name}.glb"),
                                  export_format="GLB", use_selection=True)
        prop.location = home

    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / "props.blend"), compress=True)
    (args.out_dir / "props.json").write_text(json.dumps(
        {"source": str(args.source), "straighten": args.straighten,
         "yaw_threshold": args.yaw_threshold, "props": record}, indent=2))
    print(f"SPLIT:: wrote {len(props)} props to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
