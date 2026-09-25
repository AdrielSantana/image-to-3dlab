"""Tests for the retopology + texture-transfer step.

The bpy-free parts are the ones that quietly ruin a 10-minute bake: an out-of-range face
target that re-fragments the atlas we are trying to fix, and a ray distance that either
misses the original surface entirely (empty atlas) or reaches across the body and samples
the far side.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_retopo_bake.py"


def _load():
    spec = importlib.util.spec_from_file_location("retopo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retopo = _load()


def test_defaults_are_sensible():
    source, dest, faces, size, angle, voxel, *_surface = retopo.parse_args(["--", "a.glb", "b.glb"])
    assert (source, dest) == ("a.glb", "b.glb")
    assert faces == 20000
    assert size == 2048
    assert angle == pytest.approx(math.radians(89.0))
    assert 0.0005 <= voxel <= 0.05


def test_explicit_arguments_are_honoured():
    _, _, faces, size, angle, voxel, *_surface = retopo.parse_args(
        ["--", "a.glb", "b.glb", "8000", "4096", "60", "0.003"]
    )
    assert faces == 8000
    assert size == 4096
    assert angle == pytest.approx(math.radians(60.0))
    assert voxel == pytest.approx(0.003)


def test_zero_voxel_size_means_skip_the_remesh():
    """0 is an escape hatch, not an invalid size.

    The voxel pass was adopted because the raw mesh looked hopelessly non-manifold, a
    reading that came from measuring it unwelded — 43.7% against 0.63% welded, same file.
    It costs the creases, so going direct has to be expressible.
    """
    _, _, _faces, _size, _angle, voxel, *_surface = retopo.parse_args(
        ["--", "a.glb", "b.glb", "40000", "2048", "89", "0"]
    )
    assert voxel == 0.0


def test_rejects_a_voxel_size_coarse_enough_to_melt_the_subject():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "20000", "2048", "89", "0.5"])


def test_rejects_a_voxel_size_fine_enough_to_exhaust_memory():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "20000", "2048", "89", "0.00001"])


def test_rejects_a_face_target_that_would_refragment_the_atlas():
    """The whole point is fewer, larger UV islands; 500k quads defeats it."""
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "500000"])


def test_rejects_a_face_target_too_low_to_hold_a_silhouette():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "50"])


def test_rejects_a_bad_atlas_size():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "20000", "3000"])


def test_missing_arguments_exit_with_usage():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "only.glb"])


# --- ray distance --------------------------------------------------------------------


def test_ray_distance_scales_with_the_asset():
    """A fixed distance would miss entirely on a small asset and cross-sample on a big one."""
    small = retopo.ray_distance((0.1, 0.2, 0.15))
    large = retopo.ray_distance((10.0, 20.0, 15.0))
    assert large == pytest.approx(small * 100)


def test_ray_distance_uses_the_largest_dimension():
    assert retopo.ray_distance((1.0, 5.0, 2.0)) == pytest.approx(retopo.ray_distance((5.0, 5.0, 5.0)))


def test_ray_distance_is_a_small_fraction_not_the_whole_body():
    """Reaching across the body would sample the far side's colour onto the near side."""
    assert retopo.ray_distance((1.0, 1.0, 1.0)) < 0.1


def test_quadriflow_reduced_detects_a_silent_refusal():
    """QuadriFlow declines with a Blender *warning*, not an exception.

    When its preconditions are unmet the operator leaves the mesh untouched and the script
    carries on. On the Snag that shipped 1.3M triangles from a request for 20,000. An
    unchanged count is the only reliable signal that it did not run.
    """
    assert retopo.quadriflow_reduced(662_328, 20_112, 20_000) is True
    assert retopo.quadriflow_reduced(662_328, 662_328, 20_000) is False


def test_quadriflow_reduced_rejects_a_result_nowhere_near_the_target():
    # Changed, but still vastly larger than asked for: not a retopology.
    assert retopo.quadriflow_reduced(662_328, 400_000, 20_000) is False


def test_quadriflow_reduced_tolerates_approximation():
    # QuadriFlow approximates the target rather than hitting it, so nearby counts pass.
    assert retopo.quadriflow_reduced(600_000, 24_000, 20_000) is True
    assert retopo.quadriflow_reduced(600_000, 59_000, 20_000) is True


def test_surface_knobs_have_neutral_organic_defaults():
    """Only base colour is baked, so these stand in for a metallic-roughness map.

    Defaults are a neutral organic surface rather than Blender's metallic 0 / roughness
    0.5, which reads as dead plastic under any light.
    """
    *_, metallic, roughness, ior = retopo.parse_args(["--", "a.glb", "b.glb"])
    assert (metallic, roughness, ior) == (0.25, 0.65, 1.45)


def test_surface_knobs_are_tunable_per_asset():
    # Wet bark and dry stone want different answers, so these are arguments, not constants.
    *_, metallic, roughness, ior = retopo.parse_args(
        ["--", "a.glb", "b.glb", "20000", "2048", "89", "0.004", "0.648", "0.686", "1.4"]
    )
    assert (metallic, roughness, ior) == (0.648, 0.686, 1.4)


@pytest.mark.parametrize("bad", [
    ["--", "a.glb", "b.glb", "20000", "2048", "89", "0.004", "1.5"],
    ["--", "a.glb", "b.glb", "20000", "2048", "89", "0.004", "0.5", "-0.2"],
    ["--", "a.glb", "b.glb", "20000", "2048", "89", "0.004", "0.5", "0.5", "9.0"],
])
def test_surface_knobs_reject_impossible_values(bad):
    with pytest.raises(SystemExit):
        retopo.parse_args(bad)

def test_the_decimate_ratio_is_computed_against_triangles_not_quads():
    """The bug this fixes doubled every face target this repo has ever set.

    Blender's COLLAPSE decimation applies its ratio to triangles; the voxel remesh before
    it emits quads. Measured on the Pixal3D fox: 200,632 quads = 401,264 triangles, asked
    for 40,000, got 79,991 — almost exactly twice.
    """
    quads = 200632
    triangles = quads * 2

    ratio = retopo.decimate_ratio(40000, triangles)
    assert triangles * ratio == pytest.approx(40000, rel=1e-6)

    # The old calculation, kept here as the thing that must not come back.
    wrong = min(1.0, 40000 / quads)
    assert triangles * wrong == pytest.approx(80000, rel=1e-6)


def test_the_ratio_never_exceeds_one():
    """Asking for more faces than exist must not inflate the mesh."""
    assert retopo.decimate_ratio(100000, 5000) == 1.0


def test_a_degenerate_mesh_does_not_divide_by_zero():
    assert retopo.decimate_ratio(40000, 0) == 1.0


def test_normal_map_is_off_unless_asked_for():
    assert retopo.wants_normal_map(["blender", "--", "a.glb", "b.glb"]) is False
    assert retopo.wants_normal_map(["blender", "--", "a.glb", "b.glb", "--normal-map"]) is True
    # Before the separator it is Blender's argument, not ours.
    assert retopo.wants_normal_map(["blender", "--normal-map", "--", "a.glb", "b.glb"]) is False


def test_normal_map_flag_leaves_the_positional_order_alone():
    parsed = retopo.parse_args(["--", "a.glb", "b.glb", "5000", "--normal-map", "1024"])
    source, dest, faces, size, *_ = parsed
    assert (source, dest, faces, size) == ("a.glb", "b.glb", 5000, 1024)
    assert len(parsed) == 9


def test_an_unknown_option_is_refused_rather_than_read_as_a_path():
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "--normals"])


def _glb(normals, tangents):
    """The smallest GLB carrying one primitive's normals and tangents."""
    import json
    import struct

    import numpy as np

    normals = np.asarray(normals, dtype="<f4")
    tangents = np.asarray(tangents, dtype="<f4")
    binary = normals.tobytes() + tangents.tobytes()
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": normals.nbytes},
            {"buffer": 0, "byteOffset": normals.nbytes, "byteLength": tangents.nbytes},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": len(tangents), "type": "VEC4"},
        ],
        "meshes": [{"primitives": [{"attributes": {"NORMAL": 0, "TANGENT": 1}}]}],
    }
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), retopo.GLB_JSON) + text
    body += struct.pack("<II", len(binary), retopo.GLB_BIN) + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body, 20 + len(text) + 8


def _tangents(glb, bin_start, count):
    import numpy as np

    return np.frombuffer(glb, dtype="<f4", count=count * 4,
                         offset=bin_start + count * 12).reshape(count, 4)


def test_a_zero_tangent_gets_a_unit_direction_across_its_normal():
    import numpy as np

    normals = [[0, 0, 1], [0.6, 0, 0.8], [1, 0, 0]]
    tangents = [[1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    glb, bin_start = _glb(normals, tangents)
    repaired, count = retopo.repair_zero_tangents(glb)
    assert count == 2
    assert len(repaired) == len(glb)
    fixed = _tangents(repaired, bin_start, 3)
    assert fixed[0].tolist() == [1, 0, 0, 1]          # a good tangent is left alone
    for i in (1, 2):
        assert np.linalg.norm(fixed[i, :3]) == pytest.approx(1.0, abs=1e-6)
        assert np.dot(fixed[i, :3], normals[i]) == pytest.approx(0.0, abs=1e-6)
        assert fixed[i, 3] == 1.0


def test_a_glb_without_zero_tangents_comes_back_untouched():
    glb, _ = _glb([[0, 0, 1]], [[1, 0, 0, 1]])
    repaired, count = retopo.repair_zero_tangents(glb)
    assert count == 0
    assert repaired is glb


def test_something_that_is_not_a_glb_is_refused():
    with pytest.raises(ValueError):
        retopo.repair_zero_tangents(b"PK\x03\x04" + bytes(40))
