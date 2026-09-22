"""Tests for the text-to-image step in the viewer.

A generation run costs minutes, so everything cheap is checked here: that the command line
is the measured-fast one rather than the upstream default, that a prompt cannot smuggle
arguments, that progress lines parse, and that the non-commercial licence reaches the
sidecar. The pure functions are imported, never re-implemented.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

VIEWER = Path(__file__).resolve().parents[1] / "viewer"


def _load():
    sys.path.insert(0, str(VIEWER))
    spec = importlib.util.spec_from_file_location("image_api", VIEWER / "image_api.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


api = _load()
WEIGHTS = {
    "diffusion_model": Path("/w/dit.gguf"),
    "llm": Path("/w/llm.gguf"),
    "vae": Path("/w/vae.safetensors"),
}


def command_for(prompt="a fox", **overrides):
    settings = api.clean_settings(overrides)
    return api.build_command(prompt, settings, Path("/out/x.png"), WEIGHTS,
                             binary=Path("/bin/sd-cli"))


def test_defaults_are_the_measured_fast_ones():
    """cfg 6.0 (the upstream recipe) makes the model do two passes per step for nothing:
    17 minutes against 11 for an identical picture. 768/10 steps was 4m22s."""
    assert api.DEFAULTS["cfg_scale"] == 1.0
    assert api.DEFAULTS["steps"] == 10
    assert api.DEFAULTS["width"] == api.DEFAULTS["height"] == 768


def test_offload_to_cpu_is_not_used():
    """Measured slower AND higher peak RAM on unified memory, despite the upstream recipe
    recommending it. Do not reinstate without re-measuring."""
    assert "--offload-to-cpu" not in command_for()


def test_command_carries_all_three_weights():
    command = command_for()
    for flag, key in (("--diffusion-model", "diffusion_model"), ("--llm", "llm"),
                      ("--vae", "vae")):
        assert command[command.index(flag) + 1] == str(WEIGHTS[key])


def test_prompt_is_one_argument_not_shell_text():
    """The command is a list handed to Popen without a shell, so a prompt containing
    quotes, semicolons or flags stays a prompt."""
    nasty = 'a fox"; rm -rf / #  --steps 500 --seed 9'
    command = command_for(nasty)
    assert command[command.index("-p") + 1] == nasty
    assert command.count("--steps") == 1
    assert command[command.index("--steps") + 1] == "10"


def test_negative_prompt_only_appears_when_asked():
    assert "--negative-prompt" not in command_for()
    command = command_for(negative_prompt="blurry")
    assert command[command.index("--negative-prompt") + 1] == "blurry"
    # and the positive prompt survives the insertion
    assert command[command.index("-p") + 1] == "a fox"


@pytest.mark.parametrize("given,expected", [(700, 704), (769, 768), (10, 256), (9000, 1536)])
def test_sizes_are_snapped_to_a_multiple_of_32(given, expected):
    """sd.cpp fails deep in a run on a size it cannot use. A user typing 700 means 'about
    this big', so round rather than refuse."""
    assert api.clean_settings({"width": given})["width"] == expected
    assert expected % api.SIZE_STEP == 0


@pytest.mark.parametrize("field,given", [("steps", "lots"), ("cfg_scale", None),
                                         ("seed", "abc"), ("width", {})])
def test_rubbish_settings_fall_back_to_defaults(field, given):
    assert api.clean_settings({field: given})[field] == api.DEFAULTS[field]


def test_steps_are_capped():
    assert api.clean_settings({"steps": 9999})["steps"] == api.MAX_STEPS
    assert api.clean_settings({"steps": 0})["steps"] == 1


def test_unknown_sampler_falls_back():
    assert api.clean_settings({"sampler": "wishful"})["sampler"] == api.DEFAULTS["sampler"]
    assert api.clean_settings({"sampler": "heun"})["sampler"] == "heun"


def test_progress_line_parses_with_an_eta():
    event = api.parse_progress("  |=========>    | 7/10 - 13.38s/it")
    assert event["step"] == 7 and event["total_steps"] == 10
    assert event["percent"] == 70.0
    assert event["eta_seconds"] == pytest.approx(3 * 13.38, rel=1e-3)


def test_decode_and_completion_lines_parse():
    assert api.parse_progress("decode_first_stage completed, taking 133.59s")["phase"] == "decoding"
    finished = api.parse_progress("generate_image completed in 262.30s")
    assert finished["phase"] == "finished"
    assert finished["generate_seconds"] == 262.30


def test_ordinary_chatter_is_not_an_event():
    for line in ("[INFO ] loading model", "", "ggml_metal_init: found device"):
        assert api.parse_progress(line) is None


def test_slug_is_filesystem_safe():
    name = api.slug("Glitchkin Hummingbird: a sleek/fantasy bird!! 100%")
    assert "/" not in name and ":" not in name and "%" not in name
    assert name and not name.startswith("-") and not name.endswith("-")


def test_slug_survives_a_prompt_with_nothing_usable():
    assert api.slug("!!!???") == "image"


def test_provenance_records_the_non_commercial_restriction():
    """A PNG in a folder six months from now remembers nothing on its own."""
    record = api.provenance("a fox", api.clean_settings({}), 262.3, Path("/o/fox.png"))
    assert record["license"]["classification"] == "research-only"
    assert "Built with Qwen" == record["license"]["attribution"]
    assert "inherits" in record["license"]["inherited_by_derivatives"]
    assert record["prompt"] == "a fox"
    json.dumps(record)  # must survive being written to the sidecar


def test_output_goes_to_a_research_only_folder():
    """Restricted pictures are separated by folder, so nobody has to open a sidecar to
    find out which ones they are."""
    assert api.OUTPUT_ROOT.name == "research_only"


def test_only_one_job_runs_at_a_time(tmp_path):
    manager = api.ImageJobManager(output_root=tmp_path)
    first = manager.create("one", api.clean_settings({}))
    first.status = "running"
    with pytest.raises(RuntimeError, match="already being generated"):
        manager.create("two", api.clean_settings({}))
    manager.finish(first)
    assert manager.create("two", api.clean_settings({})) is not None


def test_a_finished_job_frees_the_slot(tmp_path):
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("one", api.clean_settings({}))
    job.status = "done"
    manager.finish(job)
    assert manager.create("two", api.clean_settings({})) is not None


def test_missing_weights_names_what_is_missing():
    exc = api.MissingWeights(["qwen_image_2.1-Q8_0.gguf"])
    assert "qwen_image_2.1-Q8_0.gguf" in str(exc)
    assert "Setup & Status" in str(exc)


def test_job_describe_hides_the_result_until_it_exists(tmp_path):
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("one", api.clean_settings({}))
    assert job.describe()["result_url"] is None
    job.status = "done"
    assert job.describe()["result_url"].endswith("/result.png")


def test_sidecar_names_each_weight_file_readably():
    """The first sidecar written recorded a Python tuple's repr as the value, which is
    not something a person opening the file six months later can use."""
    weights = api.provenance("a fox", api.clean_settings({}), 1.0,
                             Path("/o/x.png"))["model"]["weights"]
    for key in ("diffusion_model", "llm", "vae"):
        assert set(weights[key]) == {"cache_dir", "file"}
        assert weights[key]["file"].endswith((".gguf", ".safetensors"))
        assert "(" not in weights[key]["cache_dir"]
