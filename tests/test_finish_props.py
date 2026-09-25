"""Tests for finishing prop-sheet props into compressed LODs.

The run is several Blender bakes per prop, so the parts that decide what gets baked are
checked here: the LOD face counts, which files are picked up, where each LOD is written,
and that gltfpack only compresses. A gltfpack call that also simplified would undo the
reason every LOD is re-baked from the original.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finish_props.py"


def _load():
    spec = importlib.util.spec_from_file_location("finish_props", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finish = _load()


def test_the_default_lods_are_most_detailed_first():
    assert finish.parse_lods("5000,2500,1000") == [5000, 2500, 1000]
    assert finish.parse_lods(",".join(map(str, finish.DEFAULT_LODS))) == list(finish.DEFAULT_LODS)


def test_one_lod_is_enough():
    assert finish.parse_lods("8000") == [8000]


@pytest.mark.parametrize("text", ["", "abc", "5000,500", "5000,250000", "1000,2500", "3000,3000"])
def test_bad_lods_are_refused(text):
    with pytest.raises(SystemExit):
        finish.parse_lods(text)


def test_a_directory_yields_its_glbs_in_order(tmp_path):
    for name in ("crate.glb", "barrel.glb", "props.json", "props.blend"):
        (tmp_path / name).write_bytes(b"x")
    assert finish.collect_props(tmp_path) == [tmp_path / "barrel.glb", tmp_path / "crate.glb"]


def test_a_single_glb_is_accepted(tmp_path):
    prop = tmp_path / "chest.glb"
    prop.write_bytes(b"x")
    assert finish.collect_props(prop) == [prop]


def test_nothing_to_finish_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        finish.collect_props(tmp_path)
    with pytest.raises(SystemExit):
        finish.collect_props(tmp_path / "missing.glb")


def test_each_lod_has_a_plain_and_a_web_file_per_prop():
    out = Path("out")
    assert finish.lod_path(out, "chest", 0) == out / "chest" / "chest_LOD0.glb"
    assert finish.lod_path(out, "chest", 2, web=True) == out / "chest" / "chest_LOD2.web.glb"


def test_gltfpack_only_compresses(tmp_path):
    command = finish.gltfpack_command(Path("/bin/gltfpack"), Path("a.glb"), Path("b.glb"))
    assert command[:5] == ["/bin/gltfpack", "-i", "a.glb", "-o", "b.glb"]
    assert "-cc" in command and "-tw" in command
    assert not any(flag.startswith("-s") for flag in command)


def test_an_explicit_gltfpack_wins_and_must_exist(tmp_path):
    binary = tmp_path / "gltfpack"
    binary.write_bytes(b"")
    assert finish.find_gltfpack(binary, which=lambda _: "/usr/bin/gltfpack") == binary
    with pytest.raises(SystemExit):
        finish.find_gltfpack(tmp_path / "nope")


def test_gltfpack_is_found_on_path_then_in_vendor(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "REPO", tmp_path)
    assert finish.find_gltfpack(which=lambda _: "/usr/bin/gltfpack") == Path("/usr/bin/gltfpack")
    assert finish.find_gltfpack(which=lambda _: None) is None
    vendored = tmp_path / "vendor" / "gltfpack" / "gltfpack"
    vendored.parent.mkdir(parents=True)
    vendored.write_bytes(b"")
    assert finish.find_gltfpack(which=lambda _: None) == vendored


def test_every_lod_is_baked_with_a_normal_map():
    command = finish.retopo_command(Path("in.glb"), Path("out.glb"), 1000, 1024,
                                    finish.ANGLE, finish.VOXEL, 0.25, 0.65, 1.45,
                                    normal_map=True)
    assert command[-1] == "--normal-map"
