"""Tests for the Pixal3D wrapper.

Two things decide whether a run is even comparable to our others: whether the image goes in
pre-matted (which skips BiRefNet and keeps the cutout identical to the TRELLIS runs), and
whether the gauge camera is passed at all — Pixal3D conditions on pixel-aligned features
projected through it, so a missing FOV is not a cosmetic difference.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pixal3d_generate.py"


def _load():
    spec = importlib.util.spec_from_file_location("pixal3d_generate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


px = _load()


def test_a_matted_image_uses_the_single_view_path():
    command = px.build_command(
        Path("fox.png"), Path("out.glb"), res=1024, seed=42, fov=px.DEFAULT_FOV,
        models=Path("/m"), cli=Path("/bin/trellis-cli"), matted=True,
    )
    assert "--sv-image" in command
    assert command[command.index("--sv-image") + 1] == "fox.png"
    assert "--bg-removal" not in command


def test_an_unmatted_image_is_handed_to_birefnet():
    command = px.build_command(
        Path("fox.jpg"), Path("out.glb"), res=1024, seed=42, fov=px.DEFAULT_FOV,
        models=Path("/m"), cli=Path("/bin/trellis-cli"), matted=False,
    )
    assert "--sv-image" not in command
    assert command[command.index("--bg-removal") + 1] == "birefnet"


def test_the_gauge_camera_and_weight_family_are_always_passed():
    command = px.build_command(
        Path("fox.png"), Path("out.glb"), 1024, 42, px.DEFAULT_FOV,
        Path("/m"), Path("/bin/trellis-cli"), True,
    )
    assert command[command.index("--fov") + 1] == str(px.DEFAULT_FOV)
    assert command[command.index("--pixal3d-weights") + 1] == "sv"
    assert command[-1] == "out.glb"  # output is positional and last


def test_the_default_fov_is_twenty_degrees():
    import math

    assert math.degrees(px.DEFAULT_FOV) == pytest.approx(20.0)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("[2/6] SS proj conditioning + flow", ("ss", 33)),
        ("[3/6] shape SLAT flow (LR 512 -> upsample -> HR 1024 cascade)", ("shape", 50)),
        ("[6/6] write out.glb", ("write", 100)),
    ],
)
def test_stage_banners_are_recognised(line, expected):
    stage, percent = px.stage_from_banner(line)
    assert stage == expected[0]
    assert percent == min(99, expected[1])


def test_the_bar_never_reaches_a_hundred_before_the_run_ends():
    _stage, percent = px.stage_from_banner("[6/6] write out.glb")
    assert percent == 99


def test_ordinary_output_is_not_a_stage():
    assert px.stage_from_banner("done in 349.9s -> out.glb") is None
    assert px.stage_from_banner("[cond] flow (load + runner + sampler) (77.6s)") is None
    assert px.stage_from_banner("ggml_metal_library_compile_pipeline: loaded kernel") is None


def test_stage_zero_is_not_a_stage():
    """`[0/6]` is the banner echoing its input, before any work happens."""
    assert px.stage_from_banner("[0/6] Pixal3D single view: fox.png") is None


def test_readiness_reports_what_is_missing(tmp_path):
    state = px.readiness(cli=tmp_path / "absent", models=tmp_path / "nope")
    assert state["ready"] is False
    assert state["cli_built"] is False
    assert state["weights_present"] == 0


def test_readiness_needs_the_whole_weight_set(tmp_path):
    cli = tmp_path / "trellis-cli"
    cli.write_text("#!/bin/sh\n")
    models = tmp_path / "models"
    models.mkdir()
    for index in range(8):
        (models / f"part{index}.gguf").write_bytes(b"x")

    assert px.readiness(cli, models)["ready"] is False  # 8 of 9

    (models / "part8.gguf").write_bytes(b"x")
    assert px.readiness(cli, models)["ready"] is True
