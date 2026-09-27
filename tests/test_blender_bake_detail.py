"""Tests for the bake-detail step: normal map + metallic-roughness onto a finished mesh.

The bpy-free parts are the ones that silently ruin an asset: baking across two meshes
that do not line up (a garbage normal map that still "succeeds"), overwriting a repaint's
own metallic-roughness map with the source's, and arguments arriving in the wrong order.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_bake_detail.py"


def _load():
    spec = importlib.util.spec_from_file_location("bake_detail", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bd = _load()


def test_arguments_come_after_the_double_dash():
    argv = ["blender", "--background", "--python", "x.py", "--",
            "high.glb", "low.glb", "out.glb", "1024"]
    args = bd.parse_args(argv)
    assert (args.source, args.target, args.output) == ("high.glb", "low.glb", "out.glb")
    assert args.size == 1024


def test_the_map_size_defaults_to_2048():
    assert bd.parse_args(["--", "a.glb", "b.glb", "c.glb"]).size == 2048


def test_too_few_arguments_is_an_error_not_a_guess():
    with pytest.raises(SystemExit):
        bd.parse_args(["--", "a.glb", "b.glb"])


def test_a_map_size_outside_the_sane_range_is_refused():
    with pytest.raises(SystemExit):
        bd.parse_args(["--", "a.glb", "b.glb", "c.glb", "100"])
    with pytest.raises(SystemExit):
        bd.parse_args(["--", "a.glb", "b.glb", "c.glb", "16384"])


def test_the_same_shape_at_a_different_scale_fits():
    """A repaint may rescale the mesh; a uniform scale is fine and gets corrected."""
    fit = bd.fit(high_size=(0.9, 0.6, 0.88), low_size=(1.8, 1.2, 1.76))
    assert fit.ok
    assert fit.scale == pytest.approx(2.0)
    assert fit.spread == pytest.approx(1.0)


def test_a_rotated_or_different_mesh_is_refused():
    """The orc bake on 2026-09-27 read a spread of 2.06 and baked garbage without complaint."""
    fit = bd.fit(high_size=(0.906, 0.876, 0.612), low_size=(0.908, 0.611, 0.879))
    assert not fit.ok
    assert fit.spread > 1.4


def test_the_real_orc_pair_is_accepted():
    """Axis ratios measured on the aligned orc bake: 1.0021, 0.9986, 1.0035."""
    fit = bd.fit(high_size=(1.0, 1.0, 1.0), low_size=(1.0021, 0.9986, 1.0035))
    assert fit.ok


def test_a_degenerate_mesh_is_refused_rather_than_divided_by():
    fit = bd.fit(high_size=(1.0, 0.0, 1.0), low_size=(1.0, 1.0, 1.0))
    assert not fit.ok


def test_metallic_roughness_is_transferred_onto_a_bare_retopo():
    assert bd.transfer_metallic_roughness(source_has_map=True, target_has_map=False)


def test_a_repaint_keeps_its_own_metallic_roughness():
    """Hunyuan's PBR repaint makes its own map, matched to the new colours."""
    assert not bd.transfer_metallic_roughness(source_has_map=True, target_has_map=True)


def test_nothing_is_transferred_when_the_source_has_no_map():
    assert not bd.transfer_metallic_roughness(source_has_map=False, target_has_map=False)


def test_the_ray_reach_scales_with_the_asset():
    assert bd.ray_reach((2.0, 1.0, 0.5)) == pytest.approx(0.04)
    assert bd.ray_reach((0.5, 0.2, 0.1)) == pytest.approx(0.01)


def test_the_sign_fix_is_the_one_from_the_normal_bake_script():
    """One implementation of the fix, imported, not a re-derived copy (test rule 1)."""
    assert bd.resolve_tangent_sign.__module__ == "blender_bake_normals"


def test_a_texel_pointing_into_the_surface_is_flipped_back_out():
    np = pytest.importorskip("numpy")
    inward = np.array([[[128, 128, 0]]], dtype=np.uint8)   # z = -1
    fixed, fraction = bd.resolve_tangent_sign(inward)
    assert fixed[0, 0, 2] > 250
    assert fraction == pytest.approx(1.0)
