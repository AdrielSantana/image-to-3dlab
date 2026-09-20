"""Tests for the browser finishing job.

Every setting here ends up as a subprocess argument, so the tests that matter are the ones
covering what a browser can send: out-of-range numbers, unknown keys, and wrong types. The
underlying scripts reject bad values with a SystemExit deep inside a Blender run, which
would surface to the user as "worker exited with code 1" several minutes later.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "viewer" / "finish_api.py"


def _load():
    spec = importlib.util.spec_from_file_location("finish_api", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finish = _load()


def test_defaults_are_used_when_nothing_is_sent():
    settings = finish.normalise_settings({})
    assert settings == finish.DEFAULT_SETTINGS
    assert settings is not finish.DEFAULT_SETTINGS  # a copy, not the shared dict


def test_client_values_override_defaults():
    settings = finish.normalise_settings({"metallic": 0.648, "roughness": 0.686, "ior": 1.4})
    assert settings["metallic"] == pytest.approx(0.648)
    assert settings["roughness"] == pytest.approx(0.686)
    assert settings["ior"] == pytest.approx(1.4)
    assert settings["faces"] == finish.DEFAULT_SETTINGS["faces"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("faces", 500), ("faces", 500000),      # blender_retopo_bake rejects both
        ("metallic", 1.5), ("roughness", -0.1),
        ("ior", 0.5), ("ior", 4.0),
        ("voxel", 0.5),                          # coarse enough to melt the subject
        ("texture_size", 8192), ("paint_steps", 0),
    ],
)
def test_out_of_range_values_are_rejected_before_a_subprocess_sees_them(key, value):
    with pytest.raises(ValueError) as caught:
        finish.normalise_settings({key: value})
    assert key in str(caught.value)


def test_a_non_numeric_value_is_rejected_with_its_key():
    with pytest.raises(ValueError) as caught:
        finish.normalise_settings({"faces": "lots"})
    assert "faces" in str(caught.value)


def test_unknown_keys_are_ignored_rather_than_forwarded():
    """A typo must not reach the command line as a stray flag."""
    settings = finish.normalise_settings({"facees": 1234, "metallic": 0.4})
    assert "facees" not in settings
    assert settings["faces"] == finish.DEFAULT_SETTINGS["faces"]
    assert settings["metallic"] == pytest.approx(0.4)


def test_integer_settings_stay_integers():
    """`--faces 40000.0` is not accepted by argparse's int type."""
    settings = finish.normalise_settings({"faces": 39999.6, "paint_res": 512.0})
    assert isinstance(settings["faces"], int)
    assert isinstance(settings["paint_res"], int)


def test_zero_voxel_is_allowed_because_it_means_skip_the_remesh():
    assert finish.normalise_settings({"voxel": 0})["voxel"] == 0


def test_the_command_carries_every_setting():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    settings = finish.normalise_settings({"metallic": 0.648, "faces": 39935})
    command = finish.build_command(job, settings)

    assert command[2].endswith("retopo_repaint.py")
    assert str(job.asset_path) in command
    assert str(job.image_path) in command
    assert str(job.result_glb) in command
    assert command[command.index("--metallic") + 1] == "0.648"
    assert command[command.index("--faces") + 1] == "39935"


def test_skip_flags_appear_only_when_asked():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    plain = finish.build_command(job, finish.normalise_settings({}))
    assert "--skip-paint" not in plain
    assert "--skip-compress" not in plain

    skipped = finish.build_command(job, finish.normalise_settings({"skip_paint": True}))
    assert "--skip-paint" in skipped


def test_stage_lines_become_progress_events():
    event = finish.stage_event("I2L_STAGE::repaint::Repainting from fox.png at 512px")
    assert event == {
        "phase": "repaint", "overall_pct": 35,
        "message": "Repainting from fox.png at 512px",
    }


def test_progress_never_goes_backwards_through_the_stages():
    phases = ["retopologise", "repaint", "compress", "done"]
    percentages = [finish.STAGE_PROGRESS[phase] for phase in phases]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100


def test_ordinary_output_is_not_mistaken_for_a_stage():
    assert finish.stage_event("[compress] 32.0 -> 4.8 MB") is None
    assert finish.stage_event("Blender quit") is None


def test_a_message_containing_the_separator_survives():
    event = finish.stage_event("I2L_STAGE::done::Finished at 4.8 MB :: all good")
    assert event["message"] == "Finished at 4.8 MB :: all good"


def test_job_ids_are_validated_before_lookup():
    manager = finish.FinishJobManager(Path("/tmp/finish-root"))
    assert manager.get("../../etc/passwd") is None
    assert manager.get("not-a-job-id") is None
    assert manager.get("0" * 32) is None  # well-formed but unknown
