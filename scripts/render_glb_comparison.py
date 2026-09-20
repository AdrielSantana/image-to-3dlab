#!/usr/bin/env python3
"""Render several GLBs from one fixed camera and lay them out as a comparison image.

For documentation: a claim like "these two settings produce the same asset" is far easier
to check against a picture than against a paragraph. This renders each GLB headlessly, so
it never touches a running Blender session, and composes the results side by side.

    python scripts/render_glb_comparison.py out.jpg \
        a.glb:"sdpa" b.glb:"MLX fp32" c.glb:"MLX fp16" [--azimuth 215] [--elevation 14]

Every panel uses the same camera, the same lights, and -- importantly -- the *same* crop
box, computed across all of them. Cropping each panel to its own subject would rescale them
independently and make identical assets look different, which is the exact failure the
image is meant to rule out.

Output is JPEG sized for a slim repository; see docs/images/README.md.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BLENDER_DEFAULT = Path("/Applications/Blender.app/Contents/MacOS/Blender")
PANEL_W, PANEL_H = 1200, 820


def bounding_sphere_radius(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> float:
    """Radius of the sphere enclosing an axis-aligned box.

    Half the longest side is not enough: an elongated subject presents its diagonal at some
    azimuths, so framing on one axis crops it at those angles and not others.
    """
    half = [(hi[i] - lo[i]) / 2 for i in range(3)]
    return math.sqrt(sum(h * h for h in half)) or 1.0


def camera_distance(radius: float, half_angle: float, margin: float = 1.02) -> float:
    """Distance at which a sphere of ``radius`` fills a camera with this half field of view."""
    if half_angle <= 0 or half_angle >= math.pi / 2:
        raise ValueError(f"half_angle must be within (0, pi/2); got {half_angle}")
    return radius / math.sin(half_angle) * margin


def shared_crop_box(boxes, size, pad: int = 18):
    """One crop box covering every panel's subject, clamped to the frame.

    Shared on purpose. Per-panel crops would scale each subject differently and
    manufacture visual differences between assets that are actually identical.
    """
    if not boxes:
        raise ValueError("need at least one subject box")
    width, height = size
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)
    return (
        max(0, left - pad), max(0, top - pad),
        min(width, right + pad), min(height, bottom + pad),
    )


def parse_asset(spec: str) -> tuple[Path, str]:
    """``path.glb:Label`` -> (path, label). The label defaults to the file stem."""
    if ":" in spec:
        head, _, label = spec.rpartition(":")
        if head and not head.endswith(".glb") and Path(spec).exists():
            return Path(spec), Path(spec).stem
        return Path(head), label
    return Path(spec), Path(spec).stem


def blender_render_code() -> str:
    """The script handed to Blender. Kept thin: the maths above is what gets tested."""
    return '''
import sys
import math
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst, azimuth, elevation, panel_w, panel_h, clay = (
    argv[0], argv[1], float(argv[2]), float(argv[3]), int(argv[4]), int(argv[5]),
    argv[6] == "1")

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("no mesh in " + src)

if clay:
    # Strip every material for a neutral clay render. Texture and geometry fail in
    # different ways, and a broken UV map makes a sound mesh look ruined -- so when the
    # question is about shape, the paint has to come off.
    neutral = bpy.data.materials.new("clay")
    neutral.use_nodes = True
    bsdf = neutral.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.62, 0.62, 0.60, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(neutral)

lo = [float("inf")] * 3
hi = [float("-inf")] * 3
for o in meshes:
    for corner in o.bound_box:
        w = o.matrix_world @ Vector(corner)
        for i in range(3):
            lo[i] = min(lo[i], w[i]); hi[i] = max(hi[i], w[i])
centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
radius = math.sqrt(sum(((hi[i] - lo[i]) / 2) ** 2 for i in range(3))) or 1.0

world = bpy.data.worlds.new("W")
bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.04, 0.04, 0.045, 1)

scene = bpy.context.scene
scene.render.resolution_x = panel_w
scene.render.resolution_y = panel_h

cam_data = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam
cam_data.sensor_fit = "AUTO"
half = min(cam_data.angle_x, cam_data.angle_y) / 2
dist = radius / math.sin(half) * 1.02

a, e = math.radians(azimuth), math.radians(elevation)
cam.location = (centre[0] + dist * math.cos(e) * math.sin(a),
                centre[1] - dist * math.cos(e) * math.cos(a),
                centre[2] + dist * math.sin(e))
cam.rotation_euler = (Vector(centre) - cam.location).to_track_quat("-Z", "Y").to_euler()

for name, offset, energy in (("key", (2.5, -2.0, 2.4), 6.0),
                             ("fill", (-2.6, -1.4, 0.7), 2.2),
                             ("rim", (0.4, 2.8, 1.8), 3.2)):
    light = bpy.data.lights.new(name, type="AREA")
    light.energy = energy * (radius ** 2) * 40
    light.size = radius * 2
    obj = bpy.data.objects.new(name, light)
    obj.location = tuple(centre[i] + offset[i] * radius for i in range(3))
    obj.rotation_euler = (Vector(centre) - obj.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(obj)

available = {i.identifier for i in type(scene.render).bl_rna.properties["engine"].enum_items}
for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
    if candidate in available:
        scene.render.engine = candidate
        break

scene.render.image_settings.file_format = "PNG"
scene.render.filepath = dst
bpy.ops.render.render(write_still=True)
'''


def compose(panels, output: Path, width: int, quality: int) -> Path:
    from PIL import Image, ImageChops, ImageDraw

    loaded = [(label, Image.open(path).convert("RGB")) for label, path in panels]

    def subject_box(im):
        bg = Image.new("RGB", im.size, im.getpixel((2, 2)))
        mask = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 12 else 0)
        return mask.getbbox() or (0, 0, *im.size)

    box = shared_crop_box([subject_box(im) for _, im in loaded], loaded[0][1].size)
    crops = [(label, im.crop(box)) for label, im in loaded]

    w, h = crops[0][1].size
    bar, gap = 44, 6
    sheet = Image.new("RGB", (w * len(crops) + gap * (len(crops) - 1), h + bar), (18, 18, 21))
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(crops):
        x = i * (w + gap)
        sheet.paste(im, (x, bar))
        draw.text((x + 14, 14), label, fill=(233, 233, 237))

    sheet = sheet.resize((width, int(sheet.height * width / sheet.width)), Image.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=quality, optimize=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="JPEG to write")
    parser.add_argument("assets", nargs="+", help="each as path.glb or path.glb:Label")
    parser.add_argument("--azimuth", type=float, default=215.0)
    parser.add_argument("--elevation", type=float, default=14.0)
    parser.add_argument("--width", type=int, default=1800, help="final image width")
    parser.add_argument("--quality", type=int, default=76)
    parser.add_argument("--clay", action="store_true",
                        help="strip materials and render neutral clay, so the comparison "
                             "is about shape rather than paint")
    parser.add_argument("--blender", type=Path, default=BLENDER_DEFAULT)
    args = parser.parse_args(argv)

    blender = args.blender if args.blender.exists() else Path(shutil.which("blender") or "")
    if not blender or not blender.exists():
        raise SystemExit(f"Blender not found at {args.blender}; pass --blender")

    parsed = [parse_asset(spec) for spec in args.assets]
    missing = [str(p) for p, _ in parsed if not p.is_file()]
    if missing:
        raise SystemExit(f"missing GLB(s): {', '.join(missing)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        script = tmp / "render.py"
        script.write_text(blender_render_code())
        panels = []
        for index, (path, label) in enumerate(parsed):
            png = tmp / f"panel_{index}.png"
            # check=False on purpose: Blender can exit non-zero having still written a
            # usable frame, and can exit zero having written nothing. The file is the
            # only honest success signal, so that is what is tested below.
            result = subprocess.run(
                [str(blender), "--background", "--python", str(script), "--",
                 str(path), str(png), str(args.azimuth), str(args.elevation),
                 str(PANEL_W), str(PANEL_H), "1" if args.clay else "0"],
                capture_output=True, text=True, check=False,
            )
            if not png.is_file():
                sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
                raise SystemExit(f"render failed for {path}")
            panels.append((label, png))
            print(f"rendered {label}", flush=True)
        out = compose(panels, args.output, args.width, args.quality)

    print(f"{out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
