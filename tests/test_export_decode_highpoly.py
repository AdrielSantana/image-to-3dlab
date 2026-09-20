"""Tests for turning a cached decode into a bake source.

The bake reads the source mesh's *normals*, so the two properties that decide whether the
resulting normal map is usable are that the mesh gets smaller and that it comes out facing
outward. Both are checked here on meshes small enough to reason about.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_decode_highpoly.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_decode_highpoly", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ed = _load()


def _sphere(subdivisions: int = 4):
    return trimesh.creation.icosphere(subdivisions=subdivisions)


def test_decimate_reduces_to_about_the_target():
    pytest.importorskip("fast_simplification")
    sphere = _sphere()
    target = len(sphere.faces) // 4

    _vertices, faces = ed.decimate(sphere.vertices, sphere.faces, target)
    assert len(faces) <= target * 1.1
    assert len(faces) < len(sphere.faces)


def test_decimate_leaves_a_mesh_smaller_than_the_target_alone():
    """Asking for more faces than exist must not invent any, or resample the mesh."""
    sphere = _sphere(subdivisions=1)
    vertices, faces = ed.decimate(sphere.vertices, sphere.faces, 1_000_000)
    assert len(faces) == len(sphere.faces)
    assert len(vertices) == len(sphere.vertices)


def test_target_faces_zero_keeps_every_face():
    sphere = _sphere(subdivisions=1)
    _vertices, faces = ed.decimate(sphere.vertices, sphere.faces, 0)
    assert len(faces) == len(sphere.faces)


def test_prepare_orients_an_inside_out_component():
    """Two spheres, one inverted — exactly the shape of the decoder's defect.

    Without this the bake would read inverted normals across that whole component and
    write a patch of the map that lights backwards.
    """
    good = _sphere(subdivisions=2)
    bad = _sphere(subdivisions=2)
    bad.apply_translation([5, 0, 0])
    bad.faces = np.fliplr(bad.faces)
    combined = trimesh.util.concatenate([good, bad])

    mesh, stats = ed.prepare_highpoly(
        combined.vertices, combined.faces, target_faces=0
    )
    assert stats["components_welded"] == 2
    assert stats["inverted_components_before"] == 1
    assert stats["inverted_components_after"] == 0
    assert mesh.volume > 0


def test_prepare_welds_seam_split_vertices_before_anything_else():
    """The decoder splits vertices along attribute seams; nothing works until they weld."""
    sphere = _sphere(subdivisions=2)
    split = sphere.copy()
    split.unmerge_vertices()
    assert len(split.vertices) == len(split.faces) * 3

    mesh, stats = ed.prepare_highpoly(split.vertices, split.faces, target_faces=0)
    assert stats["faces_welded"] == len(sphere.faces)
    assert len(mesh.vertices) == len(sphere.vertices)
    assert stats["components_welded"] == 1


def test_prepare_reports_the_face_count_at_each_step():
    pytest.importorskip("fast_simplification")
    sphere = _sphere()
    target = len(sphere.faces) // 4

    _mesh, stats = ed.prepare_highpoly(
        sphere.vertices, sphere.faces, target_faces=target
    )
    assert stats["faces_in"] == len(sphere.faces)
    assert stats["faces_decimated"] <= target * 1.1
    assert stats["faces_out"] == stats["faces_decimated"]
    assert len(stats["extents"]) == 3


def test_prepare_does_not_decimate_unless_asked():
    """The default must ship every face: decimating this mesh shatters it."""
    sphere = _sphere(subdivisions=3)
    _mesh, stats = ed.prepare_highpoly(sphere.vertices, sphere.faces)
    assert stats["faces_out"] == len(sphere.faces)
    assert "faces_decimated" not in stats


def test_component_count_sees_separate_pieces():
    """The count that exposes a simplifier shattering the mesh it was asked to reduce."""
    pytest.importorskip("scipy")
    one = _sphere(subdivisions=1)
    other = _sphere(subdivisions=1)
    other.apply_translation([5, 0, 0])
    pair = trimesh.util.concatenate([one, other])

    assert ed.component_count(one.vertices, one.faces) == 1
    assert ed.component_count(pair.vertices, pair.faces) == 2


def test_prepare_reports_components_either_side_of_decimation():
    pytest.importorskip("fast_simplification")
    pytest.importorskip("scipy")
    sphere = _sphere(subdivisions=3)

    _mesh, stats = ed.prepare_highpoly(
        sphere.vertices, sphere.faces, target_faces=len(sphere.faces) // 4
    )
    assert stats["components_welded"] == 1
    assert "components_decimated" in stats


def test_no_repair_skips_the_orientation_pass():
    """--no-repair is an explicit escape hatch, so it must really leave winding alone."""
    sphere = _sphere(subdivisions=2)
    flipped = sphere.copy()
    flipped.faces = np.fliplr(flipped.faces)

    mesh, stats = ed.prepare_highpoly(
        flipped.vertices, flipped.faces, target_faces=0, repair=False
    )
    assert "inverted_components_after" not in stats
    assert mesh.volume < 0
