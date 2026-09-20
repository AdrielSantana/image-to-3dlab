"""Tests for the retopologise → repaint → compress chain runner.

The whole point of the script is that every asset gets identical treatment, so the things
worth testing are the ones that would silently differ per asset: the stage list, and the
positional argument order handed to `blender_retopo_bake.py`. That script takes nine
positional arguments after `--`, and swapping two of them produces a differently-tuned
asset rather than an error — exactly the kind of failure a comparison would hide.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retopo_repaint.py"


def _load():
    spec = importlib.util.spec_from_file_location("retopo_repaint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load()


def test_the_full_chain_runs_three_stages_in_order():
    assert rr.stage_plan(skip_paint=False, skip_compress=False) == [
        "retopologise", "repaint", "compress",
    ]


def test_skipping_paint_keeps_the_transferred_texture_path():
    assert rr.stage_plan(skip_paint=True, skip_compress=False) == [
        "retopologise", "compress",
    ]


def test_skipping_compress_leaves_the_asset_uncompressed():
    assert rr.stage_plan(skip_paint=False, skip_compress=True) == [
        "retopologise", "repaint",
    ]


def test_retopology_is_never_skipped():
    """It is what makes the repaint affordable; there is no route that omits it."""
    assert rr.stage_plan(skip_paint=True, skip_compress=True) == ["retopologise"]


def test_the_blender_command_passes_arguments_in_the_documented_order():
    command = rr.retopo_command(
        Path("in.glb"), Path("out.glb"), faces=40000, atlas=2048, angle=89.0,
        voxel=0.004, metallic=0.648, roughness=0.686, ior=1.4,
        blender=Path("/bin/blender"),
    )
    separator = command.index("--")
    positional = command[separator + 1:]
    assert positional == [
        "in.glb", "out.glb", "40000", "2048", "89.0", "0.004", "0.648", "0.686", "1.4",
    ]


def test_the_blender_command_runs_headless_with_the_retopo_script():
    command = rr.retopo_command(
        Path("in.glb"), Path("out.glb"), 40000, 2048, 89.0, 0.004, 0.25, 0.65, 1.45,
    )
    assert "--background" in command
    assert command[command.index("--python") + 1].endswith("blender_retopo_bake.py")
    # Headless, not the live socket: a Cycles bake through the GUI blocks the viewport.
    assert "--python-console" not in command


def test_a_zero_voxel_fraction_is_passed_through_not_dropped():
    """0 means 'skip the remesh' downstream; it must survive as a value."""
    command = rr.retopo_command(
        Path("in.glb"), Path("out.glb"), 40000, 2048, 89.0, 0.0, 0.25, 0.65, 1.45,
    )
    assert command[command.index("--") + 6] == "0.0"
