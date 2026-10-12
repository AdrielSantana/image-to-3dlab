"""Tests for scripts/bootstrap_front_back.py: consent, prerequisites, and what it fetches."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "viewer")]

import backend_catalog
import bootstrap_front_back as boot


@pytest.fixture
def routes_ready(monkeypatch):
    monkeypatch.setattr(boot, "missing_routes", list)


def test_the_announcement_names_both_downloads_their_sizes_and_licences():
    text = boot.announcement()
    assert "5.9 GB" in text and "0.75 GB" in text and "6.6 GB in all" in text
    assert "mmproj" in text and "pixal3d-q8_0-v1" in text
    assert "DINOv3" in text and "Qwen" in text


def test_nothing_downloads_without_a_yes_and_nobody_to_ask(monkeypatch, routes_ready):
    monkeypatch.setattr(boot.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(boot.pixal3d, "install_multiview", lambda *a: pytest.fail("downloaded"))
    monkeypatch.setattr(boot.qwen, "install_vision", lambda *a: pytest.fail("downloaded"))
    assert boot.main([]) == 1


def test_a_yes_fetches_both(monkeypatch, routes_ready):
    fetched = []
    monkeypatch.setattr(boot.pixal3d, "install_multiview", lambda *a: fetched.append("mv"))
    monkeypatch.setattr(boot.qwen, "install_vision", lambda *a: fetched.append("vision"))
    assert boot.main(["--yes"]) == 0 and fetched == ["mv", "vision"]


def test_missing_routes_stop_it_before_anything_is_fetched(monkeypatch, capsys):
    monkeypatch.setattr(boot, "missing_routes", lambda: ["Pixal3D (python scripts/bootstrap_pixal3d.py)"])
    monkeypatch.setattr(boot.pixal3d, "install_multiview", lambda *a: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    assert "Pixal3D" in capsys.readouterr().out


def test_the_catalogue_entry_quotes_the_same_sizes():
    entry = backend_catalog.BY_ID["front-back"]
    assert [w.bytes_expected for w in entry.weights] == [boot.MV_BYTES, boot.VISION_BYTES]
    flows = entry.weights[0]
    assert flows.pattern == "*_mv.gguf"  # measured without the files linked from single-view
    assert boot.TOTAL_GB == pytest.approx(boot.pixal3d.MV_GB + boot.qwen.VISION[2], abs=0.1)
    vision = next(w for w in entry.weights if "mmproj" in (w.pattern or ""))
    assert vision.pattern.endswith(boot.qwen.VISION[1])
