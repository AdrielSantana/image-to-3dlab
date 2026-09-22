"""Tests for the text-to-image bootstrap.

The download is 13.4 GB, so what matters here is everything that happens *before* one
byte moves: that the user is told what is coming, that nothing starts without a yes, and
that a non-interactive run cannot silently spend someone's bandwidth.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "bootstrap_qwen_image", SCRIPTS / "bootstrap_qwen_image.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


boot = _load()


def test_announcement_names_backend_route_size_and_licence():
    """AGENTS.md: a download path must name the backend, the route and the size, and
    require an affirmative answer. The first three are checked here."""
    text = boot.announcement()
    assert "Qwen-Image 2.1" in text
    assert "stable-diffusion.cpp" in text
    assert "13.4 GB" in text
    assert "NON-COMMERCIAL" in text
    assert "Built with Qwen" in text


def test_announcement_lists_every_file_with_its_size():
    text = boot.announcement()
    for _, filename, size in boot.WEIGHTS:
        assert filename in text
        assert f"{size:.2f}" in text


def test_total_matches_the_catalogue_figure():
    """The Setup page and this script must not quote different numbers for one download."""
    assert boot.total_gb() == pytest.approx(13.4, abs=0.05)


def test_build_only_does_not_promise_weights():
    text = boot.announcement(weights=False)
    assert "weights:" not in text
    assert "build:" in text


def test_nothing_downloads_without_a_yes(monkeypatch, capsys):
    """Declining must stop, and must not call either installer."""
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(boot.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert boot.main([]) == 1
    assert "Nothing downloaded" in capsys.readouterr().out


def test_a_non_interactive_run_refuses_rather_than_hanging(monkeypatch, capsys):
    """No tty and no --yes means nobody is there to consent. Waiting on input() would
    hang a CI job or a subprocess forever; defaulting to yes would break the rule."""
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(boot.sys.stdin, "isatty", lambda: False)
    assert boot.main([]) == 1
    assert "--yes" in capsys.readouterr().out


def test_yes_proceeds_without_asking(monkeypatch):
    called = []
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: called.append("build"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: called.append("weights"))
    monkeypatch.setattr(boot, "binary_present", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("should not ask"))
    assert boot.main(["--yes"]) == 0
    assert called == ["build", "weights"]


def test_weights_only_skips_the_binary(monkeypatch):
    called = []
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: called.append("build"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: called.append("weights"))
    assert boot.main(["--yes", "--weights-only"]) == 0
    assert called == ["weights"]


def test_an_existing_binary_is_not_redownloaded(monkeypatch, capsys):
    """Re-running a bootstrap on an installed tree must no-op, not re-fetch. AGENTS.md
    asks patch scripts to be idempotent and the same courtesy applies here."""
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: pytest.fail("redownloaded"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: None)
    monkeypatch.setattr(boot, "binary_present", lambda: True)
    assert boot.main(["--yes"]) == 0
    assert "already installed" in capsys.readouterr().out


@pytest.mark.parametrize("name,expected", [
    ("sd-master-74988b2-bin-Darwin-macOS-26.6.2-arm64.zip", True),
    ("sd-master-74988b2-bin-Darwin-macOS-15.0-arm64.zip", True),
    ("sd-master-74988b2-bin-win-avx2-x64.zip", False),
    ("sd-master-74988b2-bin-Darwin-macOS-13-x64.zip", False),
    ("source.tar.gz", False),
])
def test_release_asset_is_matched_by_substring_not_exact_name(name, expected):
    """The asset name carries upstream's build-machine OS version, which changes without
    warning. Matching it exactly would break the installer on their next CI upgrade."""
    picked = boot.pick_asset([{"name": name}])
    assert (picked is not None) is expected


def test_asset_matching_prefers_nothing_over_the_wrong_architecture():
    assert boot.pick_asset([{"name": "sd-bin-Darwin-macOS-14-x64.zip"}]) is None
