"""The SF3D installer: Mac and NVIDIA from one script, gated weights fetched only on a yes."""

from __future__ import annotations

import io

import backend_catalog
import bootstrap_sf3d as boot
import pytest


def test_weight_total_matches_the_catalogue():
    """Including DINOv2, which SF3D otherwise downloads silently on its first run."""
    sf3d = next(b for b in backend_catalog.CATALOG if b.id == "sf3d")
    assert sum(w.bytes_expected for w in sf3d.weights) == pytest.approx(
        boot.total_gb() * backend_catalog.GB, rel=0.02)
    assert {w.source for w in sf3d.weights} == {repo for repo, _, _ in boot.WEIGHTS}


@pytest.mark.parametrize("key,cuda_env,metal_env", [
    ("macos-arm64", "0", "1"),
    ("linux-nvidia", "1", "0"),
])
def test_extension_build_flags_follow_the_machine(monkeypatch, key, cuda_env, metal_env):
    monkeypatch.setattr(boot, "find_nvcc", lambda: "/usr/local/cuda/bin/nvcc")
    env = boot.build_env(key, {})
    assert env["USE_CUDA"] == cuda_env and env["USE_METAL"] == metal_env


def test_linux_without_nvcc_builds_the_cpu_baker(monkeypatch):
    """The texture baker compiles its CUDA kernel only with a toolkit present; without one
    it still builds, and bakes on the CPU."""
    monkeypatch.setattr(boot, "find_nvcc", lambda: None)
    assert boot.build_env("linux-nvidia", {})["USE_CUDA"] == "0"


def test_nvcc_off_path_is_put_on_it(monkeypatch):
    monkeypatch.setattr(boot, "find_nvcc", lambda: "/usr/local/cuda/bin/nvcc")
    env = boot.build_env("linux-nvidia", {"PATH": "/usr/bin"})
    assert env["PATH"].split(":")[0] == "/usr/local/cuda/bin"


def test_announcement_names_backend_route_size_and_gating(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(boot, "find_nvcc", lambda: None)
    text = boot.announcement()
    for needle in ("Stable Fast 3D", "4.9 GB", "facebook/dinov2-large", "gated",
                   "Stability AI Community License"):
        assert needle in text


@pytest.mark.parametrize("key", [None, "windows-nvidia"])
def test_unsupported_machines_are_refused_before_anything(monkeypatch, capsys, key):
    monkeypatch.setattr(boot, "target", lambda: key)
    monkeypatch.setattr(boot, "install_code", lambda *a: pytest.fail("installed"))
    monkeypatch.setattr(boot, "install_weights", lambda: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    assert "Nothing downloaded" in capsys.readouterr().out


def test_no_yes_and_no_terminal_means_no_download(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(boot.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(boot, "install_code", lambda *a: pytest.fail("installed"))
    monkeypatch.setattr(boot, "install_weights", lambda: pytest.fail("downloaded"))
    assert boot.main([]) == 1


def test_code_and_weights_halves(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    called = []
    monkeypatch.setattr(boot, "install_code", lambda *a: called.append("code"))
    monkeypatch.setattr(boot, "install_weights", lambda: called.append("weights"))
    assert boot.main(["--yes", "--code-only"]) == 0
    assert boot.main(["--yes", "--weights-only"]) == 0
    assert boot.main(["--yes"]) == 0
    assert called == ["code", "weights", "code", "weights"]


def test_gated_weights_explain_how_to_get_access(monkeypatch, capsys):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")

    def refused():
        raise boot.GatedAccess("stabilityai/stable-fast-3d")

    monkeypatch.setattr(boot, "install_weights", refused)
    assert boot.main(["--yes", "--weights-only"]) == 1
    out = capsys.readouterr().out
    assert "huggingface.co/stabilityai/stable-fast-3d" in out and "hf auth login" in out


def test_the_viewer_can_run_it():
    import download_api

    command = download_api.COMMANDS["sf3d"]
    assert command[1].endswith("bootstrap_sf3d.py") and "--yes" in command
