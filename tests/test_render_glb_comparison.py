"""Tests for the headless GLB comparison renderer.

Only the pure parts are tested, which is the point of extracting them: the framing maths
and the layout decisions are what can silently produce a misleading image, while the
Blender call itself is a thin shell.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_glb_comparison.py"
SPEC = importlib.util.spec_from_file_location("render_glb_comparison", SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_bounding_sphere_covers_the_diagonal_not_the_longest_side():
    """An elongated subject turns its diagonal to the camera at some angles.

    Sizing on the longest side alone crops it at those azimuths and not others, which makes
    a comparison image lie about the assets rather than about the camera.
    """
    radius = mod.bounding_sphere_radius((0, 0, 0), (10, 2, 2))
    assert radius == pytest.approx(math.sqrt(25 + 1 + 1))
    assert radius > 10 / 2


def test_bounding_sphere_never_returns_zero():
    # A degenerate (flat or empty) box would otherwise put the camera inside the subject.
    assert mod.bounding_sphere_radius((1, 1, 1), (1, 1, 1)) == 1.0


def test_camera_distance_frames_the_sphere():
    # At exactly radius/sin(half) the sphere touches the frame edge; the margin backs off.
    exact = 5.0 / math.sin(0.5)
    assert mod.camera_distance(5.0, 0.5, margin=1.0) == pytest.approx(exact)
    assert mod.camera_distance(5.0, 0.5, margin=1.1) == pytest.approx(exact * 1.1)


@pytest.mark.parametrize("half_angle", [0.0, -0.3, math.pi / 2, 2.0])
def test_camera_distance_rejects_impossible_fields_of_view(half_angle):
    with pytest.raises(ValueError):
        mod.camera_distance(1.0, half_angle)


def test_shared_crop_box_spans_every_panel():
    box = mod.shared_crop_box([(100, 50, 200, 300), (80, 70, 210, 280)], (400, 400), pad=0)
    assert box == (80, 50, 210, 300)


def test_shared_crop_box_is_clamped_to_the_frame():
    box = mod.shared_crop_box([(2, 2, 398, 398)], (400, 400), pad=50)
    assert box == (0, 0, 400, 400)


def test_shared_crop_box_rejects_an_empty_set():
    with pytest.raises(ValueError):
        mod.shared_crop_box([], (10, 10))


def test_asset_spec_splits_path_and_label():
    path, label = mod.parse_asset("output/a.glb:MLX fp16")
    assert path == Path("output/a.glb")
    assert label == "MLX fp16"


def test_asset_spec_without_a_label_uses_the_file_stem():
    path, label = mod.parse_asset("output/storm_ram.glb")
    assert path == Path("output/storm_ram.glb")
    assert label == "storm_ram"


def test_blender_code_imports_everything_it_uses():
    """The Blender-side script runs in a fresh interpreter with no inherited imports."""
    code = mod.blender_render_code()
    for name in ("import bpy", "from mathutils import Vector", "import math", "import sys"):
        assert name in code


def test_blender_code_does_not_pin_a_single_render_engine():
    # The EEVEE enum has been renamed across Blender releases; pinning one name turns an
    # upgrade into a silent failure.
    code = mod.blender_render_code()
    assert "BLENDER_EEVEE_NEXT" in code and "BLENDER_EEVEE" in code and "CYCLES" in code


def test_blender_code_starts_from_an_empty_scene():
    # Otherwise a user's default cube, camera and light end up in the render.
    assert "read_factory_settings(use_empty=True)" in mod.blender_render_code()
