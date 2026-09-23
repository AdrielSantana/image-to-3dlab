"""What machine is this: the one question every bootstrap and the viewer ask first."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from image_to_3dlab import host


def _smi(returncode: int, stdout: str):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")
    return run


def test_os_family_names_the_three_we_support():
    assert host.os_family("darwin") == "macos"
    assert host.os_family("linux") == "linux"
    assert host.os_family("win32") == "windows"
    assert host.os_family("freebsd13") == "other"


def test_apple_silicon_is_decided_without_asking_for_a_gpu():
    def never(*a, **k):
        raise AssertionError("a Mac must not probe for nvidia-smi")
    assert host.host_platform("darwin", "arm64", nvidia=never) == host.APPLE


def test_intel_mac_is_not_apple_silicon():
    assert host.host_platform("darwin", "x86_64", nvidia=lambda: False) == "other"


def test_linux_and_windows_with_an_nvidia_card_are_nvidia():
    assert host.host_platform("linux", "x86_64", nvidia=lambda: True) == host.NVIDIA
    assert host.host_platform("win32", "AMD64", nvidia=lambda: True) == host.NVIDIA


def test_linux_without_a_card_is_other():
    assert host.host_platform("linux", "x86_64", nvidia=lambda: False) == "other"


def test_nvidia_gpu_needs_nvidia_smi_on_path():
    assert host.has_nvidia_gpu(which=lambda _: None) is False


def test_nvidia_gpu_needs_a_listed_gpu():
    which = lambda _: "/usr/bin/nvidia-smi"
    listed = "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-abc)\n"
    assert host.has_nvidia_gpu(which=which, run=_smi(0, listed)) is True
    # A driver installed with no card, or a container without the GPU mounted.
    assert host.has_nvidia_gpu(which=which, run=_smi(0, "")) is False
    assert host.has_nvidia_gpu(which=which, run=_smi(9, "NVIDIA-SMI has failed")) is False


def test_nvidia_smi_that_hangs_or_vanishes_means_no_gpu():
    which = lambda _: "/usr/bin/nvidia-smi"

    def hangs(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    def vanished(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    assert host.has_nvidia_gpu(which=which, run=hangs) is False
    assert host.has_nvidia_gpu(which=which, run=vanished) is False


def test_executable_gets_exe_only_on_windows(tmp_path: Path):
    assert host.executable(tmp_path, "sd-cli", "windows") == tmp_path / "sd-cli.exe"
    assert host.executable(tmp_path, "sd-cli", "linux") == tmp_path / "sd-cli"
    assert host.executable(tmp_path, "sd-cli", "macos") == tmp_path / "sd-cli"


@pytest.mark.parametrize("platform_id,family,expected", [
    ("apple-silicon", "macos", "macos-arm64"),
    ("nvidia", "linux", "linux-nvidia"),
    ("nvidia", "windows", "windows-nvidia"),
    ("other", "linux", None),
    ("other", "macos", None),
])
def test_build_target_maps_the_machine_to_a_prebuilt(platform_id, family, expected):
    assert host.build_target(platform_id, family) == expected


# The header `nvidia-smi` printed on the RunPod 4090 used for the 2026-09-23 test.
SMI_HEADER = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
"""


def test_driver_cuda_version_is_read_from_the_smi_header():
    which = lambda _: "/usr/bin/nvidia-smi"
    assert host.driver_cuda_version(which=which, run=_smi(0, SMI_HEADER)) == (12, 8)


def test_driver_cuda_version_is_none_without_a_driver_or_a_header():
    assert host.driver_cuda_version(which=lambda _: None) is None
    which = lambda _: "/usr/bin/nvidia-smi"
    assert host.driver_cuda_version(which=which, run=_smi(0, "no header here")) is None
    assert host.driver_cuda_version(which=which, run=_smi(9, SMI_HEADER)) is None


def test_compute_capability_drops_the_dot():
    """CMake wants `89`, nvidia-smi says `8.9`."""
    which = lambda _: "/usr/bin/nvidia-smi"
    assert host.compute_capability(which=which, run=_smi(0, "8.9\n")) == "89"
    assert host.compute_capability(which=which, run=_smi(0, "garbage")) is None
    assert host.compute_capability(which=lambda _: None) is None
