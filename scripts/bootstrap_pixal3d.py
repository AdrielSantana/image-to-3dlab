#!/usr/bin/env python3
"""Install Pixal3D (raven38/pixal3d.cpp): a `trellis-cli` build plus its Q8_0 weights.

Two halves, the same way the viewer tracks every backend. The **build** depends on the
machine:

- **Apple Silicon:** cloned and compiled from source, because Metal kernels need the
  local Xcode toolchain. That needs full Xcode, not just the Command Line Tools.
- **Linux or Windows with an NVIDIA card:** upstream's prebuilt CUDA 12 build. The CUDA
  runtime ships inside it, so there is no CUDA toolkit to install.

The **weights** are the single-view Q8_0 set plus the BiRefNet matting model, 8.4 GB.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_pixal3d.py            # says what it wants, then asks
    python scripts/bootstrap_pixal3d.py --yes      # for the viewer and for agents
    python scripts/bootstrap_pixal3d.py --build-only
    python scripts/bootstrap_pixal3d.py --weights-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host

VENDOR = REPO / "vendor" / "pixal3d-cpp"
BUILD = VENDOR / "build"
MODELS = VENDOR / "models" / "pixal3d-sv"
UPSTREAM = "https://github.com/raven38/pixal3d.cpp.git"
WEIGHTS_REPO = "raven38/pixal3d-sv-q8_0-v1"
MATTE_REPO = "ilintar/trellis2-gguf"
WEIGHTS_GB = 8.4

# Pinned, not "latest": every upstream release so far is a pre-release, and a prebuilt
# that has been run end to end is worth more than a newer one that has not.
PREBUILT_RELEASE = "v0.10.1-desktop-alpha"
RELEASE_API = "https://api.github.com/repos/raven38/pixal3d.cpp/releases/tags/{tag}"

# CUDA 12 rather than upstream's unversioned CUDA build (which is newer): CUDA 12 runs on
# older drivers, and the runtime is bundled either way.
PREBUILTS = {
    "linux-nvidia": ("trellis-cuda12-linux-x64.tar.gz", "~640 MB"),
    "windows-nvidia": ("trellis-cuda12-windows-x64.zip", "~610 MB"),
}

LICENCE = (
    "MIT (code and flow weights); the bundled image encoder is under the\n"
    "  DINOv3 License. https://huggingface.co/raven38/pixal3d-sv-q8_0-v1"
)

# Looked up through the module so a test can pretend to be another machine.
target = host.build_target


def route_and_size(key: str | None) -> tuple[str, str] | None:
    if key == "macos-arm64":
        return "built from source with Metal", "compiled locally, needs full Xcode"
    if key in PREBUILTS:
        name, size = PREBUILTS[key]
        return f"CUDA 12 prebuilt ({name}, {PREBUILT_RELEASE})", size
    return None


def announcement(build: bool = True, weights: bool = True) -> str:
    """Exactly what is about to be fetched, before anything is."""
    found = route_and_size(target())
    route = found[0] if found else "none for this machine"
    lines = ["", "About to install:", "", "  backend: Pixal3D (raven38/pixal3d.cpp)",
             f"  route:   {route}"]
    if build and found:
        lines.append(f"  build:   {found[1]} -> vendor/pixal3d-cpp/build/")
    if weights:
        lines.append(f"  weights: {WEIGHTS_GB:.1f} GB -> vendor/pixal3d-cpp/models/pixal3d-sv/")
        lines.append(f"             {WEIGHTS_REPO}, plus BiRefNet matting ({MATTE_REPO})")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def cli_path() -> Path:
    return host.executable(BUILD, "trellis-cli")


def build_present() -> bool:
    """A function so the idempotence check is testable without touching the disk."""
    return cli_path().exists()


def pick_prebuilt(assets: list[dict], key: str) -> dict | None:
    wanted = PREBUILTS[key][0]
    return next((a for a in assets if a.get("name") == wanted), None)


def unpack_prebuilt(archive: Path, destination: Path) -> Path:
    """Unpack a prebuilt into `destination` and return the runnable `trellis-cli`.

    The Linux tarball's library symlinks (`libcudart.so.12 -> libcudart.so.12.9.79`) must
    survive; the loader looks for the short names.
    """
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            # The "data" filter refuses paths that escape `destination`; it exists on
            # 3.11.4+ and 3.12+, and older interpreters get the plain extract.
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(destination, filter="data")
            else:
                bundle.extractall(destination)
    cli = host.executable(destination, "trellis-cli")
    if not cli.exists():
        raise SystemExit(f"{archive.name} contained no trellis-cli.")
    for name in ("trellis-cli", "trellis-server"):
        path = host.executable(destination, name)
        if path.exists() and not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def install_prebuilt(key: str) -> Path:
    print(f"Finding pixal3d.cpp release {PREBUILT_RELEASE}...", flush=True)
    try:
        url = RELEASE_API.format(tag=PREBUILT_RELEASE)
        with urllib.request.urlopen(url, timeout=30) as response:
            release = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Could not reach GitHub: {exc}") from exc
    asset = pick_prebuilt(release.get("assets", []), key)
    if asset is None:
        raise SystemExit(f"Release {PREBUILT_RELEASE} has no {PREBUILTS[key][0]}.")
    VENDOR.mkdir(parents=True, exist_ok=True)
    archive = VENDOR / asset["name"]
    print(f"Downloading {asset['name']} ({asset.get('size', 0) / 1e6:.0f} MB)...", flush=True)
    urllib.request.urlretrieve(asset["browser_download_url"], archive)
    try:
        cli = unpack_prebuilt(archive, BUILD)
    finally:
        archive.unlink(missing_ok=True)
    print(f"Installed {cli}")
    return cli


def build_from_source() -> Path:
    """The Apple Silicon path: clone, fetch submodules, compile with Metal."""
    for tool in ("cmake", "ninja", "git"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found (brew install {tool})")
    if subprocess.run(["xcrun", "--find", "metal"], capture_output=True,
                      check=False).returncode != 0:
        # Printed rather than run: both need sudo or change a system-wide setting.
        raise SystemExit(
            "Metal compiler unavailable. With full Xcode installed, this is usually:\n"
            "  sudo DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer "
            "xcodebuild -license accept\n"
            "  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer "
            "xcodebuild -downloadComponent MetalToolchain\n"
            "then re-run this script with DEVELOPER_DIR set."
        )
    if not (VENDOR / ".git").is_dir():
        print(f"Cloning {UPSTREAM}", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", UPSTREAM, str(VENDOR)], check=True)
    print("Fetching vendored ggml and friends", flush=True)
    subprocess.run(["git", "-C", str(VENDOR), "submodule", "update", "--init",
                    "--recursive"], check=True)
    print("Building (Metal is the default backend on Apple)", flush=True)
    subprocess.run(["cmake", "-S", str(VENDOR), "-B", str(BUILD), "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(BUILD), "-j"], check=True)
    if not build_present():
        raise SystemExit("The build finished without trellis-cli.")
    return cli_path()


def install_build(key: str) -> Path:
    return build_from_source() if key == "macos-arm64" else install_prebuilt(key)


def flatten_matte(models: Path) -> None:
    """`trellis-cli` looks for models flat in `--models`; BiRefNet lands in `q8/`."""
    nested = models / "q8" / "birefnet.gguf"
    if nested.exists() and not (models / "birefnet.gguf").exists():
        shutil.move(str(nested), str(models / "birefnet.gguf"))
    if (models / "q8").is_dir() and not any((models / "q8").iterdir()):
        (models / "q8").rmdir()


def install_weights(models: Path = MODELS) -> None:
    # Plain HTTP rather than Xet, as the shell bootstrap this replaced always used.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. pip install -r requirements-dev.txt"
        ) from exc
    models.mkdir(parents=True, exist_ok=True)
    print(f"\nFetching {WEIGHTS_REPO} ({WEIGHTS_GB:.1f} GB, resumable)...", flush=True)
    snapshot_download(WEIGHTS_REPO, local_dir=models, max_workers=2)
    hf_hub_download(MATTE_REPO, "q8/birefnet.gguf", local_dir=models)
    flatten_matte(models)
    print(f"  weights in {models}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    parser.add_argument("--build-only", action="store_true",
                        help="Install trellis-cli and stop, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only, assuming trellis-cli is present.")
    args = parser.parse_args(argv)

    key = target()
    if key is None:
        print("Pixal3D needs an Apple Silicon Mac, or Linux/Windows with an NVIDIA card "
              "(nvidia-smi must list it). Nothing downloaded.")
        return 1

    build = not args.weights_only
    weights = not args.build_only
    print(announcement(build=build, weights=weights))

    if not args.yes:
        # Non-interactive without --yes must not silently proceed, and must not hang
        # waiting on a stdin nobody is attached to.
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    if build:
        if build_present():
            print(f"\ntrellis-cli is already installed at {cli_path()}, leaving it alone.")
        else:
            install_build(key)
    if weights:
        install_weights()
    print("\nDone. Generate with:\n"
          "    python scripts/pixal3d_generate.py input.png output.glb --res 1024")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
