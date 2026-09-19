from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "scripts" / "blender_rebind_weights.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blender_rebind_weights", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Mesh:
    name = "Creature"

    def __init__(self, dimensions):
        self.dimensions = dimensions


def test_voxel_size_is_scale_relative():
    assert load_module().derived_voxel_size(Mesh((2, 4, 3))) == pytest.approx(0.048)


@pytest.mark.parametrize("fraction", [0, 0.0009, 0.051, 1])
def test_voxel_fraction_has_safe_bounds(fraction):
    with pytest.raises(ValueError):
        load_module().derived_voxel_size(Mesh((1, 1, 1)), fraction)


def test_weight_transfer_is_fail_closed_and_uses_a_disposable_proxy():
    source = MODULE.read_text()

    assert "ARMATURE_AUTO" in source
    assert 'data_types_verts = {"VGROUP_WEIGHTS"}' in source
    assert "POLYINTERP_NEAREST" in source
    assert "produced no non-empty weight groups" in source
    assert "bpy.data.objects.remove(proxy, do_unlink=True)" in source


def test_largest_component_picks_the_body_over_the_specks():
    """The moss fox's proxy came out as one 17,526-vertex body plus 13 loose 8-vertex
    voxel specks; bone heat solves one global system, so those specks made it singular
    and every one of the 34 groups came back empty."""
    module = load_module()
    body = [(i, i + 1) for i in range(9)]          # 10 verts, 0..9
    speck = [(10, 11), (11, 12)]                   # 3 verts, 10..12
    assert module.largest_component(13, body + speck) == set(range(10))


def test_largest_component_handles_one_island_and_isolated_vertices():
    module = load_module()
    assert module.largest_component(3, [(0, 1), (1, 2)]) == {0, 1, 2}
    assert module.largest_component(1, []) == {0}
    assert module.largest_component(0, []) == set()


def test_largest_component_breaks_ties_deterministically():
    module = load_module()
    assert module.largest_component(4, [(0, 1), (2, 3)]) == {0, 1}


def test_loose_islands_are_stripped_before_the_heat_solve():
    source = MODULE.read_text()
    assert "strip_loose_islands" in source
    assert source.index("strip_loose_islands") < source.index("ARMATURE_AUTO")
