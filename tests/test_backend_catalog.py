"""Tests for the weights catalogue behind the onboarding screen.

`AGENTS.md` forbids downloading weights before the user has confirmed what they want, and
the screen that asks them is only as honest as the numbers it shows. Both bugs below were
real, found the day the module was written, and both make the progress readout lie rather
than crash, which is why they get tests rather than a comment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "viewer" / "backend_catalog.py"


def _load():
    # Registered in sys.modules before exec: dataclasses resolves annotations through
    # sys.modules[cls.__module__], and a spec-loaded module that skips this raises
    # AttributeError on the first @dataclass.
    spec = importlib.util.spec_from_file_location("backend_catalog", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backend_catalog"] = module
    spec.loader.exec_module(module)
    return module


bc = _load()


def test_every_backend_states_a_licence_and_links_to_it():
    # The licence is the user's business, but we owe them the name and a way to read it.
    for backend in bc.CATALOG:
        assert backend.license_name, backend.id
        assert backend.license_url.startswith("https://"), backend.id


def test_the_hunyuan_territorial_restriction_is_surfaced():
    # Not editorialising, but not hiding it either: downloading these in a restricted
    # region is a harm we would be causing.
    hunyuan = bc.BY_ID["hunyuan_xiong"]
    assert hunyuan.caveat is not None
    for region in ("EU", "UK", "South Korea"):
        assert region in hunyuan.caveat


def test_exactly_one_backend_is_recommended():
    recommended = [b for b in bc.CATALOG if b.rank == 1]
    assert len(recommended) == 1
    assert recommended[0].id == "pixal3d"


def test_no_weight_set_contains_another(tmp_path):
    """Nested paths double-count, and a backend then reports 200% downloaded.

    Hit for real: the bootstrap moves BiRefNet *into* `pixal3d-sv/`, so a second entry
    pointed at the parent counted the whole backend twice.
    """
    for backend in bc.CATALOG:
        paths = [w.path for w in backend.weights]
        for outer in paths:
            for inner in paths:
                if outer is inner:
                    continue
                assert outer not in inner.parents, f"{backend.id}: {inner} sits inside {outer}"


def test_symlinked_files_are_counted_once(tmp_path):
    """The Hugging Face cache links `snapshots/` at `blobs/`, so following both doubles
    every byte and a 15 GB backend reports 30 GB."""
    blobs, snaps = tmp_path / "blobs", tmp_path / "snapshots"
    blobs.mkdir()
    snaps.mkdir()
    (blobs / "weight.bin").write_bytes(b"x" * 1000)
    (snaps / "weight.bin").symlink_to(blobs / "weight.bin")

    present, total = bc._dir_state(tmp_path)
    assert present is True
    assert total == 1000, "the symlink was followed and counted a second time"


def test_a_missing_directory_is_missing_not_an_error(tmp_path):
    assert bc._dir_state(tmp_path / "absent") == (False, 0)


def test_an_interrupted_download_reads_as_partial_not_ready():
    # A directory that exists but holds a tenth of the bytes is a failed fetch. Calling it
    # ready is how someone debugs a backend that was never fully downloaded.
    expected = 10 * bc.GB
    w = bc._weights_state
    assert w([{"bytes_present": expected, "bytes_expected": expected}]) == "ready"
    assert w([{"bytes_present": expected // 10, "bytes_expected": expected}]) == "partial"
    assert w([{"bytes_present": 0, "bytes_expected": expected}]) == "missing"


def test_a_slightly_short_download_still_counts_as_ready():
    # The size constants are measured and approximate, so the floor has to tolerate drift;
    # an exact comparison made a fully installed TRELLIS.2 report "partial" at 92%.
    expected = 10 * bc.GB
    assert bc._weights_state(
        [{"bytes_present": int(expected * 0.9), "bytes_expected": expected}]) == "ready"


def test_onboarding_is_needed_only_when_nothing_is_ready(monkeypatch, tmp_path):
    status = bc.catalog_status()
    assert status["needs_onboarding"] is (status["ready_count"] == 0)

    # A fresh clone: no weights *and* nothing built. Weights alone are not enough to
    # decide this, since a built TRELLIS with no weights is usable.
    import dataclasses
    absent = tmp_path / "nothing-here"
    monkeypatch.setattr(bc, "_dir_state", lambda path: (False, 0))
    monkeypatch.setattr(bc, "CATALOG", tuple(
        dataclasses.replace(b, build_probes=(absent,)) for b in bc.CATALOG
    ))
    empty = bc.catalog_status()
    assert empty["needs_onboarding"] is True
    assert empty["ready_count"] == 0
    assert all(b["state"] == "missing" for b in empty["backends"])
    assert all(b["action"] == "build" for b in empty["backends"])


def test_backends_are_listed_best_first():
    ranks = [b["rank"] for b in bc.catalog_status()["backends"]]
    assert ranks == sorted(ranks)


def test_sizes_are_stated_before_anything_is_fetched():
    for backend in bc.catalog_status()["backends"]:
        assert backend["bytes_expected"] > 0, backend["id"]
        assert backend["human_expected"].endswith(("MB", "GB")), backend["id"]


def test_human_bytes_reads_like_a_download_dialog():
    assert bc.human_bytes(0) == "0 B"
    assert bc.human_bytes(92 * 1024 ** 2) == "92.0 MB"
    assert bc.human_bytes(int(8.4 * bc.GB)) == "8.4 GB"


def test_trellis_setup_is_a_build_not_a_download():
    """Its bootstrap clones, patches and compiles; the weights come on first generation.

    Calling that a download makes the confirmation lie and makes byte progress meaningless
    — a healthy hour-long compile reported no growth and would have read as stalled.
    """
    assert bc.BY_ID["trellis"].setup_fetches_weights is False
    assert bc.BY_ID["pixal3d"].setup_fetches_weights is True
    assert bc.BY_ID["hunyuan_xiong"].setup_fetches_weights is True


def test_the_flag_reaches_the_browser():
    trellis = next(b for b in bc.catalog_status()["backends"] if b["id"] == "trellis")
    assert trellis["setup_fetches_weights"] is False


def test_a_built_trellis_with_no_weights_is_ready_not_missing():
    """Its bootstrap installs the code and fetches nothing; the weights come on first run.

    Reporting it as missing offered a Set up button that re-ran a finished bootstrap,
    which then died on its own already-applied patches (hit for real 2026-09-21 while
    testing against an empty HF_HOME).
    """
    none_present = [{"bytes_present": 0, "bytes_expected": 10 * bc.GB}]
    assert bc._state(none_present, built=True, setup_fetches=False) == "ready"
    assert bc._action(none_present, built=True, setup_fetches=False) == "none"


def test_an_unbuilt_backend_is_offered_a_build_whatever_its_weights():
    weights = [{"bytes_present": 10 * bc.GB, "bytes_expected": 10 * bc.GB}]
    assert bc._state(weights, built=False, setup_fetches=True) == "missing"
    assert bc._action(weights, built=False, setup_fetches=True) == "build"


def test_a_built_backend_missing_weights_is_offered_the_download():
    empty = [{"bytes_present": 0, "bytes_expected": 10 * bc.GB}]
    half = [{"bytes_present": 5 * bc.GB, "bytes_expected": 10 * bc.GB}]
    full = [{"bytes_present": 10 * bc.GB, "bytes_expected": 10 * bc.GB}]
    assert bc._action(empty, built=True, setup_fetches=True) == "download"
    assert bc._action(half, built=True, setup_fetches=True) == "resume"
    assert bc._action(full, built=True, setup_fetches=True) == "none"


def test_a_backend_with_no_declared_probes_is_never_called_unbuilt():
    assert all(b.build_present or b.build_probes for b in bc.CATALOG)
