"""What machine is this, in the vocabulary backends declare support in.

One module so the viewer and every bootstrap answer the question the same way. Two
answers matter today: an Apple Silicon Mac, or a Linux/Windows box with an NVIDIA card.
Everything else is "other", and a backend that does not list it will not offer a download.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

APPLE = "apple-silicon"
NVIDIA = "nvidia"
OTHER = "other"


def os_family(sys_platform: str | None = None) -> str:
    name = sys_platform or sys.platform
    if name == "darwin":
        return "macos"
    if name.startswith("linux"):
        return "linux"
    if name in ("win32", "cygwin"):
        return "windows"
    return OTHER


def _smi(args: list[str], which: Callable, run: Callable) -> str | None:
    """`nvidia-smi <args>` stdout, or None if it is missing, fails or hangs."""
    smi = which("nvidia-smi")
    if not smi:
        return None
    try:
        result = run([smi, *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or "") if result.returncode == 0 else None


def has_nvidia_gpu(which: Callable = shutil.which, run: Callable = subprocess.run) -> bool:
    """True when `nvidia-smi` lists at least one GPU.

    The driver ships `nvidia-smi`, so it is the cheapest honest test: no CUDA toolkit, no
    torch. Its presence alone is not enough (a container can have the tool but no card
    mounted), so it has to list a GPU too.
    """
    return "GPU" in (_smi(["-L"], which, run) or "")


def driver_cuda_version(which: Callable = shutil.which,
                        run: Callable = subprocess.run) -> tuple[int, int] | None:
    """The newest CUDA the installed driver supports, from `nvidia-smi`'s header.

    This is the driver's ceiling, not an installed toolkit. A binary compiled with a newer
    CUDA than this fails at its first kernel, not at load time.
    """
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", _smi([], which, run) or "")
    return (int(match[1]), int(match[2])) if match else None


def compute_capability(which: Callable = shutil.which,
                       run: Callable = subprocess.run) -> str | None:
    """The first card's compute capability as CMake spells it (`8.9` becomes `89`)."""
    out = _smi(["--query-gpu=compute_cap", "--format=csv,noheader"], which, run) or ""
    match = re.match(r"\s*(\d+)\.(\d+)", out)
    return f"{match[1]}{match[2]}" if match else None


@lru_cache(maxsize=1)
def _cached_nvidia() -> bool:
    return has_nvidia_gpu()


def host_platform(sys_platform: str | None = None, machine: str | None = None,
                  nvidia: Callable[[], bool] = _cached_nvidia) -> str:
    """APPLE, NVIDIA or OTHER.

    A Mac is decided from `sys.platform` and the CPU alone. Anything else asks the driver,
    once per process.
    """
    family = os_family(sys_platform)
    if family == "macos":
        return APPLE if (machine or platform.machine()) == "arm64" else OTHER
    if family in ("linux", "windows") and nvidia():
        return NVIDIA
    return OTHER


def find_nvcc() -> str | None:
    """The CUDA compiler: on PATH, or where the toolkit installs it by default. Its
    absence from PATH is normal, so the default location is worth checking."""
    found = shutil.which("nvcc")
    if found:
        return found
    default = Path("/usr/local/cuda/bin/nvcc")
    return str(default) if default.exists() else None


def executable(directory: Path, name: str, family: str | None = None) -> Path:
    """`directory/name`, spelled the way this OS spells a program."""
    suffix = ".exe" if (family or os_family()) == "windows" else ""
    return directory / f"{name}{suffix}"


def build_target(platform_id: str | None = None, family: str | None = None) -> str | None:
    """Which prebuilt a bootstrap should fetch here: `macos-arm64`, `linux-nvidia`,
    `windows-nvidia`, or None when nothing fits."""
    platform_id = platform_id or host_platform()
    family = family or os_family()
    if platform_id == APPLE:
        return "macos-arm64"
    if platform_id == NVIDIA and family in ("windows", "linux"):
        return f"{family}-nvidia"
    return None
