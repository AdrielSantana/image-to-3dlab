#!/usr/bin/env python3
"""Install the text-to-image route: a stable-diffusion.cpp binary and Qwen-Image weights.

Two halves, the same way the viewer tracks every other backend. The **build** is a
prebuilt `stable-diffusion.cpp` release binary (there is nothing to compile; it ships a
Metal build for Apple Silicon). The **weights** are three files totalling about 13.4 GB.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_qwen_image.py            # says what it wants, then asks
    python scripts/bootstrap_qwen_image.py --yes      # for the viewer and for scripts
    python scripts/bootstrap_qwen_image.py --build-only
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import stat
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor" / "sdcpp"
BINARY = VENDOR / "sd-cli"
RELEASES = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"

# (repo, filename, approximate gigabytes). These are the files the upstream Qwen-Image 2.1
# guide names, at the quantisations that were measured to be worth it: Q8 for the image
# model, Q4_K_M for the text encoder, and the VAE unquantised because it is small anyway.
WEIGHTS = [
    ("leejet/Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q8_0.gguf", 7.69),
    ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5.03),
    ("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", 0.68),
]

LICENCE = (
    "Qwen Research License: NON-COMMERCIAL USE ONLY, and it asks that you say\n"
    "  'Built with Qwen'. Anything you generate from these weights inherits that,\n"
    "  including a 3D asset made from a generated picture.\n"
    "  https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE"
)


def total_gb() -> float:
    return sum(size for _, _, size in WEIGHTS)


def announcement(build: bool = True, weights: bool = True) -> str:
    """Exactly what is about to be fetched, before anything is.

    A separate function so a test can assert the backend, the route, the size and the
    licence are all named without running a download.
    """
    lines = ["", "About to install:", "", "  backend: Qwen-Image 2.1 (text to image)",
             "  route:   GGUF via stable-diffusion.cpp, Metal on Apple Silicon"]
    if build:
        lines.append("  build:   prebuilt sd-cli release binary (~35 MB) -> vendor/sdcpp/")
    if weights:
        lines.append(f"  weights: {total_gb():.1f} GB total ->  Hugging Face cache")
        for repo, filename, size in WEIGHTS:
            lines.append(f"             {size:>5.2f} GB  {filename}  ({repo})")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def binary_present() -> bool:
    """Whether sd-cli is already installed. A function rather than an inline
    `BINARY.exists()` so the idempotence check is testable without touching the disk."""
    return BINARY.exists()


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def pick_asset(assets: list[dict], machine: str = "arm64") -> dict | None:
    """The macOS build for this architecture, from a GitHub release's asset list.

    Matched by substring rather than by an exact name because the release names carry the
    builder's OS version (`...-Darwin-macOS-26.6.2-arm64.zip`), which changes every time
    upstream's CI machine is updated and is none of our business.
    """
    for asset in assets:
        name = asset.get("name", "").lower()
        if name.endswith(".zip") and "darwin" in name and machine in name:
            return asset
    return None


def install_binary(destination: Path = VENDOR) -> Path:
    if not is_apple_silicon():
        raise SystemExit(
            "The prebuilt binary is macOS/arm64 only. On another machine, build "
            "stable-diffusion.cpp from source and put sd-cli in vendor/sdcpp/."
        )
    print("Finding the latest stable-diffusion.cpp release...")
    try:
        with urllib.request.urlopen(RELEASES, timeout=30) as response:
            release = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Could not reach GitHub: {exc}") from exc
    asset = pick_asset(release.get("assets", []))
    if asset is None:
        raise SystemExit(
            f"Release {release.get('tag_name')} has no macOS arm64 build. "
            "Build from source and put sd-cli in vendor/sdcpp/."
        )
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "release.zip"
    print(f"Downloading {asset['name']} ({asset.get('size', 0) / 1e6:.0f} MB)...")
    urllib.request.urlretrieve(asset["browser_download_url"], archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink(missing_ok=True)
    binary = destination / "sd-cli"
    if not binary.exists():
        found = next(destination.rglob("sd-cli"), None)
        if found is None:
            raise SystemExit("The release archive contained no sd-cli binary.")
        # Some releases nest everything one directory down; flatten so the path the
        # catalogue probes is the path that exists.
        for item in found.parent.iterdir():
            shutil.move(str(item), str(destination / item.name))
        binary = destination / "sd-cli"
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed {binary}")
    return binary


def install_weights() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. pip install -r requirements-dev.txt"
        ) from exc
    for repo, filename, size in WEIGHTS:
        print(f"\nFetching {filename} ({size:.2f} GB) from {repo}...", flush=True)
        path = hf_hub_download(repo_id=repo, filename=filename)
        print(f"  {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for scripts.")
    parser.add_argument("--build-only", action="store_true",
                        help="Install the binary and stop, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only, assuming sd-cli is already present.")
    args = parser.parse_args(argv)

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
        if binary_present():
            print(f"\nsd-cli is already installed at {BINARY}, leaving it alone.")
        else:
            install_binary()
    if weights:
        install_weights()
    print("\nDone. Open the viewer's Generate Image tab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
