"""Tests for the weight-download runner.

The parts worth testing are the ones a user reads when something goes wrong: the progress
line, the stall verdict, and the sentence shown instead of "exited with code 1". The
subprocess itself is not exercised here; it downloads gigabytes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "viewer" / "download_api.py"


def _load():
    spec = importlib.util.spec_from_file_location("download_api", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_api"] = module
    spec.loader.exec_module(module)
    return module


dl = _load()


def test_progress_bars_are_stripped_to_their_last_frame():
    # tqdm redraws with \r, so everything before the last frame is a stale repaint that
    # would otherwise pile up in the browser log.
    assert dl.strip_ansi("10%|##        |\r50%|#####     |\r90%|######### |") == "90%|######### |"
    assert dl.strip_ansi("\x1b[32mdone\x1b[0m") == "done"
    assert dl.strip_ansi("  plain line  ") == "plain line"


def test_rate_needs_two_samples_before_it_claims_anything():
    assert dl.rate_and_eta([], 100) == (None, None)
    assert dl.rate_and_eta([(0.0, 0)], 100) == (None, None)


def test_rate_and_eta_are_measured_over_the_window():
    rate, eta = dl.rate_and_eta([(0.0, 0), (10.0, 1000)], remaining=2000)
    assert rate == 100.0
    assert eta == 20.0


def test_a_window_with_no_growth_reports_no_eta():
    rate, eta = dl.rate_and_eta([(0.0, 500), (10.0, 500)], remaining=2000)
    assert rate == 0.0
    assert eta is None


def _backend():
    return dl.BY_ID["pixal3d"]


def test_progress_never_claims_completion_before_the_process_exits():
    # The expected size is an estimate, so a download that overshoots it must not show
    # 100% while the process is still running; only a clean exit says done.
    event = dl.describe_progress(_backend(), present=10 ** 13, rate=1.0, eta=1.0, stalled=False)
    assert event["overall_pct"] == 99


def test_a_stall_is_named_rather_than_shown_as_slow_progress():
    event = dl.describe_progress(_backend(), present=1000, rate=0.0, eta=None, stalled=True)
    assert "stalled" in event["detail"]
    assert event["stalled"] is True


def test_progress_states_both_numbers_not_just_a_percentage():
    # The percentage inherits the catalogue's approximate sizes; the raw bytes do not.
    event = dl.describe_progress(_backend(), present=2 * 1024 ** 3, rate=None, eta=None,
                                 stalled=False)
    assert "2.0 GB of" in event["detail"]


def test_a_gated_repo_is_explained_as_a_login_problem():
    message = dl._explain(1, ["Traceback", "401 Client Error: Unauthorized for url"])
    assert "login" in message and "retry" in message


def test_a_full_disk_is_explained_as_a_full_disk():
    assert "disk space" in dl._explain(1, ["OSError: [Errno 28] No space left on device"])


def test_a_network_failure_is_explained_as_one():
    assert "network" in dl._explain(1, ["ConnectionError: Max retries exceeded"])


def test_an_unrecognised_failure_still_quotes_the_last_line():
    message = dl._explain(2, ["something specific went wrong"])
    assert "code 2" in message and "something specific went wrong" in message


def test_every_catalogued_backend_either_has_a_command_or_says_it_has_none():
    # A Download button that 500s is worse than one that explains itself.
    for backend_id in dl.BY_ID:
        if backend_id in dl.COMMANDS:
            continue
        try:
            dl.start(backend_id)
        except RuntimeError as exc:
            assert "no automated setup" in str(exc)
        else:
            raise AssertionError(f"{backend_id} started without a command")


def test_an_unknown_backend_is_rejected():
    import pytest
    with pytest.raises(KeyError):
        dl.start("not-a-backend")


def test_the_hunyuan_command_pins_one_model_rather_than_all_three():
    # Without --model the downloader fetches every shape checkpoint: 23 GB where the
    # default route needs 5.
    command = dl.COMMANDS["hunyuan_xiong"]
    assert "--model" in command
    assert command[command.index("--model") + 1] == "2.0"


def test_removal_refuses_paths_outside_the_repo_and_cache(tmp_path, monkeypatch):
    """The guard that stops a bad catalogue entry deleting something else.

    Nothing in the shipped catalogue points outside, which is exactly why this needs a
    test: the day one does, it must fail loudly rather than run `rmtree` on it.
    """
    import dataclasses

    import pytest
    stray = tmp_path / "somewhere-else"
    stray.mkdir()
    (stray / "w.bin").write_bytes(b"x")
    real = dl.BY_ID["pixal3d"]
    # WeightSet is frozen, so build a stand-in rather than mutating the shipped one.
    rogue = dataclasses.replace(
        real, weights=(dataclasses.replace(real.weights[0], path=stray),),
    )
    monkeypatch.setitem(dl.BY_ID, "rogue", rogue)

    with pytest.raises(RuntimeError, match="refusing to delete"):
        dl.remove("rogue")
    assert stray.is_dir(), "the guard let a delete through"


def test_paths_inside_the_repo_or_cache_are_allowed():
    assert dl._inside_known_roots(dl.REPO / "vendor" / "x") is True
    assert dl._inside_known_roots(dl.HF_HUB_DIR / "models--a--b") is True
    assert dl._inside_known_roots(Path("/etc")) is False


def test_removing_an_unknown_backend_is_rejected():
    import pytest
    with pytest.raises(KeyError):
        dl.remove("not-a-backend")


def test_removal_is_refused_while_that_backend_is_downloading(monkeypatch):
    import pytest
    run = dl.DownloadRun(dl.BY_ID["pixal3d"])
    run.status = "running"
    monkeypatch.setitem(dl.DOWNLOADS, "pixal3d", run)
    with pytest.raises(RuntimeError, match="downloading right now"):
        dl.remove("pixal3d")


# --- Starting a command that is not there -----------------------------------------------
# A Windows user reported "[WinError 2] The system cannot find the file specified" after
# logging into Hugging Face. `Popen` had been handed `.venv/bin/python`, which on Windows
# is spelled `.venv\\Scripts\\python.exe`, and the raw OSError went straight to the browser.


def test_the_hunyuan_command_uses_this_os_s_interpreter_path():
    program = Path(dl.COMMANDS["hunyuan_xiong"][0])
    assert program.name in ("python", "python.exe")
    assert program.parent.name in ("bin", "Scripts")


def test_a_missing_venv_is_explained_rather_than_reported_as_errno_2(tmp_path):
    absent = tmp_path / "shape" / ".venv" / "bin" / "python"
    message = dl._missing_executable(str(absent))
    assert message is not None
    assert "uv sync" in message
    assert str(tmp_path / "shape") in message


def test_a_missing_program_names_itself():
    assert "not-a-real-program" in dl._missing_executable("not-a-real-program")


def test_a_command_that_exists_is_not_second_guessed(tmp_path):
    assert dl._missing_executable(sys.executable) is None
    assert dl._missing_executable("sh") is None or os.name == "nt"


def test_an_unsupported_machine_is_refused_before_anything_downloads(monkeypatch):
    """The refusal belongs here too: the API is what spends the bandwidth."""
    import backend_catalog

    monkeypatch.setattr(backend_catalog, "host_platform", lambda: "other")
    with pytest.raises(RuntimeError) as raised:
        dl.start("hunyuan_xiong")
    assert "Apple Silicon" in str(raised.value)
    assert "Nothing has been downloaded" in str(raised.value)
    assert "hunyuan_xiong" not in dl.DOWNLOADS
