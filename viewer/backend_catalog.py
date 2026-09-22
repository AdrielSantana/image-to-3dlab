"""What each backend needs on disk, so nothing is ever downloaded by surprise.

`AGENTS.md` makes this a rule: no script, bootstrap, viewer button or agent tool may fetch
weights until the user has confirmed which pipeline and which route. A rule needs numbers
to be honest about, and until now the viewer had none: `/api/backends` described stages
and settings but never said that picking TRELLIS.2 means ~16 GB from Hugging Face.

This module is that missing half. It is deliberately pure data plus pure functions over a
filesystem, with no imports from `generate_api`, so the onboarding screen, the CLI
bootstraps and the tests can all read the same catalogue without a circular import.

**Sizes are approximate and measured, not authoritative.** They come from a real install
on 2026-09-21 (`du -sh`) and exist to set expectations before a download, not to verify
one. Treat a mismatch as a stale constant, never as a failure.

**Licence text here is a pointer, not advice.** Each entry names the licence and links to
the original; reading it is the user's business. The one exception is a restriction with
real teeth, such as Hunyuan's territorial limits, which is surfaced as a caveat because
silently downloading those weights in a restricted region is a harm we would be causing.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
HF_HUB_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"

GB = 1024 ** 3

# Which machines a backend can run on. Apple Silicon is the only answer today -- MLX, the
# Metal kernels and the shell bootstraps all assume it -- but NVIDIA support is coming, so
# this is per-backend data rather than one "is this a Mac?" test. Adding a CUDA route later
# means adding a string to that backend's `runs_on`, not unpicking a platform check.
APPLE = "apple-silicon"
NVIDIA = "nvidia"
PLATFORM_LABELS = {APPLE: "an Apple Silicon Mac", NVIDIA: "an NVIDIA GPU"}


def venv_python(project: Path) -> Path:
    """The interpreter inside a project's `.venv`, named the way this OS names it.

    Windows puts it in `Scripts/python.exe`, everywhere else it is `bin/python`. Hardcoding
    the POSIX spelling is what turned a Windows visit into "[WinError 2] The system cannot
    find the file specified" with no clue as to which file.
    """
    if os.name == "nt":
        return project / ".venv" / "Scripts" / "python.exe"
    return project / ".venv" / "bin" / "python"


def host_platform() -> str:
    """What this machine is, in the vocabulary backends declare support in.

    Deliberately cheap and structural -- `sys.platform` and the CPU architecture, no driver
    probing. The question here is only "could this backend run here at all", asked before
    offering someone a multi-gigabyte download; whether a specific toolchain is present is
    the bootstrap's business. When the CUDA route lands, detecting it belongs in here.
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return APPLE
    return "other"


def host_label(host: str) -> str:
    """This machine, named the way its owner would name it."""
    if host in PLATFORM_LABELS:
        return PLATFORM_LABELS[host].removeprefix("an ").removeprefix("a ")
    return f"{platform.system() or 'this machine'} ({platform.machine()})"


def runs_on_phrase(backend: Backend) -> str:
    return " or ".join(PLATFORM_LABELS.get(p, p) for p in backend.runs_on)


@dataclass(frozen=True)
class WeightSet:
    """One downloadable unit, and where it lands."""

    label: str
    source: str
    bytes_expected: int
    path: Path
    note: str | None = None

    def describe(self) -> dict[str, Any]:
        present, actual = _dir_state(self.path)
        return {
            "label": self.label,
            "source": self.source,
            "source_url": f"https://huggingface.co/{self.source}",
            "bytes_expected": self.bytes_expected,
            "human_expected": human_bytes(self.bytes_expected),
            "present": present,
            "bytes_present": actual,
            "human_present": human_bytes(actual),
            "path": str(self.path),
            "note": self.note,
        }


@dataclass(frozen=True)
class Backend:
    """One generation route, as the onboarding table presents it."""

    id: str
    label: str
    best_for: str
    tradeoff: str
    license_name: str
    license_url: str
    weights: tuple[WeightSet, ...]
    install: str
    rank: int | None = None
    setup_minutes: int | None = None
    caveat: str | None = None
    # Whether running this backend's setup actually fetches the weights. TRELLIS's
    # bootstrap does not: it clones, patches and builds the Metal port, and the weights
    # arrive lazily on the first generation run. The distinction changes what the
    # confirmation says and whether byte progress means anything.
    setup_fetches_weights: bool = True
    # Files that prove the code side is installed: a compiled binary, a venv interpreter.
    # Weights and build are independent, and conflating them offers "Set up" to someone
    # who already has the build, which re-runs a bootstrap that then fails on its own
    # already-applied patches (hit for real 2026-09-21).
    build_probes: tuple[Path, ...] = ()
    extra_steps: tuple[str, ...] = field(default_factory=tuple)
    # The machines this route works on. Default rather than per-entry because every route
    # is Apple-only today; the day one of them runs on CUDA, it says so here.
    runs_on: tuple[str, ...] = (APPLE,)
    # What this route produces. Everything here made a mesh until Qwen-Image arrived, and a
    # text-to-image step sits one stage upstream of the rest of the pipeline: it is for
    # people who do not have a source image yet. The page groups on this rather than
    # guessing from the label.
    kind: str = "3d"

    @property
    def bytes_expected(self) -> int:
        return sum(w.bytes_expected for w in self.weights)

    @property
    def build_present(self) -> bool:
        """True when nothing is declared, so a weights-only backend is never 'unbuilt'."""
        return all(p.exists() for p in self.build_probes)

    def runs_here(self, host: str | None = None) -> bool:
        return (host or host_platform()) in self.runs_on

    def describe(self, host: str | None = None) -> dict[str, Any]:
        weights = [w.describe() for w in self.weights]
        present = sum(w["bytes_present"] for w in weights)
        built = self.build_present
        supported = self.runs_here(host)
        return {
            "build_present": built,
            "supported_here": supported,
            "requires": runs_on_phrase(self),
            # Said once, in words, so the screen can explain instead of a button failing.
            "platform_note": None if supported else (
                f"Needs {runs_on_phrase(self)}. Setting it up on this machine would "
                f"download gigabytes and then fail, so the button is off."
            ),
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "rank": self.rank,
            "recommended": self.rank == 1,
            "best_for": self.best_for,
            "tradeoff": self.tradeoff,
            "license": {"name": self.license_name, "url": self.license_url},
            "caveat": self.caveat,
            "install": self.install,
            "setup_minutes": self.setup_minutes,
            "setup_fetches_weights": self.setup_fetches_weights,
            "extra_steps": list(self.extra_steps),
            "weights": weights,
            "bytes_expected": self.bytes_expected,
            "human_expected": human_bytes(self.bytes_expected),
            "bytes_present": present,
            "human_present": human_bytes(present),
            "state": "unsupported" if not supported else
                     _state(weights, built, self.setup_fetches_weights),
            "action": "none" if not supported else
                      _action(weights, built, self.setup_fetches_weights),
            "percent_present": _percent(present, self.bytes_expected),
        }


# Ranked the way the README ranks them, because two orderings of the same advice is one
# too many. Rank 1 is what a newcomer should pick.
CATALOG: tuple[Backend, ...] = (
    Backend(
        id="pixal3d",
        label="Pixal3D (C++/GGML, Metal)",
        rank=1,
        best_for="Best results we have. One pass, ~6 min, no repaint needed.",
        tradeoff="Needs Xcode's Metal compiler, not just the command-line tools.",
        license_name="MIT (code + flow weights); DINOv3 License (bundled encoder)",
        license_url="https://huggingface.co/raven38/pixal3d-sv-q8_0-v1",
        install="scripts/bootstrap_pixal3d_cpp.sh",
        setup_minutes=20,
        build_probes=(REPO / "vendor" / "pixal3d-cpp" / "build" / "trellis-cli",),
        weights=(
            # One entry, not two: the bootstrap moves BiRefNet *into* pixal3d-sv/, so a
            # second set pointed at the parent directory would count everything twice.
            WeightSet("Pixal3D single-view Q8_0, with BiRefNet matting",
                      "raven38/pixal3d-sv-q8_0-v1", int(8.4 * GB),
                      REPO / "vendor" / "pixal3d-cpp" / "models" / "pixal3d-sv",
                      note="Includes the BiRefNet matting model (ilintar/trellis2-gguf), "
                           "used to cut out a subject when the image has no alpha."),
        ),
    ),
    Backend(
        id="hunyuan_xiong",
        label="Hunyuan3D-MLX (Xiong, full pipeline)",
        rank=2,
        best_for="Fast, clean results, and the quickest to run from a fresh clone.",
        tradeoff="Shape and paint are separate venvs; RealESRGAN super-res is a manual step.",
        license_name="MIT (code); Tencent Hunyuan Community License (weights)",
        license_url="https://huggingface.co/tencent/Hunyuan3D-2.1",
        install="uv sync + hunyuan_mlx/download_weights.py",
        setup_minutes=25,
        build_probes=(venv_python(REPO / "hunyuan_mlx" / "shape"),
                      venv_python(REPO / "hunyuan_mlx" / "paint")),
        caveat=(
            "The Hunyuan weights are not licensed for use in the EU, the UK or South Korea. "
            "Check the licence before downloading."
        ),
        extra_steps=(
            "RealESRGAN super-res weights are a separate conversion step; see "
            "docs/hunyuan-mlx-recipes.md.",
        ),
        weights=(
            WeightSet("Hunyuan3D-2 shape (default route)", "tencent/Hunyuan3D-2",
                      int(5.0 * GB), REPO / "hunyuan_mlx" / "shape" / "weights" / "Hunyuan3D-2"),
            WeightSet("Hunyuan3D-2.1 paint (PBR)", "tencent/Hunyuan3D-2.1", int(8.3 * GB),
                      REPO / "hunyuan_mlx" / "paint" / "weights"),
        ),
    ),
    Backend(
        id="trellis",
        label="TRELLIS.2 (clean port)",
        rank=3,
        best_for="Highest fidelity, closest to the official demo.",
        tradeoff=(
            "Slowest, and its material model bleaches flat or vector-style illustrations. "
            "Prefer photographs or softly lit 3D-style references."
        ),
        license_name="MIT (code + weights); DINOv3 License (image encoder)",
        license_url="https://huggingface.co/microsoft/TRELLIS.2-4B",
        install="viewer",
        setup_minutes=60,
        setup_fetches_weights=False,
        build_probes=(venv_python(REPO / "vendor" / "trellis-space-mac"),),
        weights=(
            WeightSet("TRELLIS.2-4B", "microsoft/TRELLIS.2-4B", int(14.0 * GB),
                      HF_HUB_DIR / "models--microsoft--TRELLIS.2-4B"),
            WeightSet("DINOv3 image encoder", "facebook/dinov3-vitl16-pretrain-lvd1689m",
                      int(1.1 * GB),
                      HF_HUB_DIR / "models--facebook--dinov3-vitl16-pretrain-lvd1689m"),
            WeightSet("TinyCLIP input advisor", "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
                      92 * 1024 ** 2,
                      HF_HUB_DIR / "models--wkcn--TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
                      note="Advisory only. Generation works without it."),
        ),
    ),
    Backend(
        id="qwen-image",
        label="Qwen-Image 2.1 (text to image)",
        kind="image",
        best_for="Makes the source image when you do not have one. Prompt in, picture out.",
        tradeoff=(
            "Non-commercial licence, and the restriction carries into any 3D asset you "
            "make from the picture."
        ),
        license_name="Qwen Research License (non-commercial)",
        license_url="https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
        install="Prebuilt stable-diffusion.cpp binary in vendor/sdcpp/",
        setup_minutes=15,
        build_probes=(REPO / "vendor" / "sdcpp" / "sd-cli",),
        caveat=(
            "The Qwen Research License is non-commercial only and asks that you say "
            "'Built with Qwen'. Anything you generate from one of these images inherits "
            "that, including after a Hunyuan repaint."
        ),
        weights=(
            WeightSet("Qwen-Image 2.1 diffusion model (Q8_0)",
                      "leejet/Qwen-Image-2.1-GGUF", int(7.69 * GB),
                      HF_HUB_DIR / "models--leejet--Qwen-Image-2.1-GGUF"),
            WeightSet("Qwen3-VL-8B text encoder (Q4_K_M)",
                      "Qwen/Qwen3-VL-8B-Instruct-GGUF", int(5.03 * GB),
                      HF_HUB_DIR / "models--Qwen--Qwen3-VL-8B-Instruct-GGUF",
                      note="Qwen-Image reads your prompt with a vision-language model, "
                           "which is why the text encoder is this large."),
            WeightSet("Qwen-Image 2.1 VAE", "Comfy-Org/Qwen-Image-2.1", int(0.68 * GB),
                      HF_HUB_DIR / "models--Comfy-Org--Qwen-Image-2.1",
                      note="Turns the generated latent back into pixels."),
        ),
    ),
)

BY_ID = {backend.id: backend for backend in CATALOG}


def catalog_status(host: str | None = None) -> dict[str, Any]:
    """The whole catalogue merged with what is actually on disk, for this machine.

    The host is reported alongside the backends because "nothing is installed" and "nothing
    can be installed here" look identical in a list of states, and only one of them is
    worth a download button.
    """
    host = host or host_platform()
    backends = [backend.describe(host) for backend in sorted(CATALOG, key=_rank_key)]
    ready = [b for b in backends if b["state"] == "ready"]
    runnable = [b for b in backends if b["supported_here"]]
    return {
        "schema_version": 1,
        # The onboarding screen exists for exactly this condition, so the server decides
        # it rather than leaving each client to re-derive the rule.
        "needs_onboarding": not ready,
        "ready_count": len(ready),
        "host": {
            "id": host,
            "label": host_label(host),
            "any_backend_runs_here": bool(runnable),
            "supported": sorted({PLATFORM_LABELS.get(p, p)
                                 for b in CATALOG for p in b.runs_on}),
        },
        "backends": backends,
    }


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _dir_state(path: Path) -> tuple[bool, int]:
    """Whether a weight directory exists, and how many bytes are in it.

    Size is what drives the progress readout during a download, so an unreadable file is
    skipped rather than raised: a partially written cache is the normal case here, not an
    error worth failing the whole status call for.
    """
    if not path.is_dir():
        return False, 0
    total = 0
    for item in path.rglob("*"):
        try:
            # Skip symlinks. The Hugging Face cache stores one copy under `blobs/` and
            # links to it from `snapshots/`, so following both counts every byte twice and
            # a 16 GB backend reports 30 GB, which makes the progress percentage nonsense.
            if item.is_symlink() or not item.is_file():
                continue
            total += item.stat().st_size
        except OSError:
            continue
    return total > 0, total


def _weights_state(weights: list[dict[str, Any]]) -> str:
    """ready, partial or missing, judged against expected size rather than mere existence.

    A directory that exists but holds a tenth of the bytes is an interrupted download, and
    calling that "ready" is how someone ends up debugging a backend that was never fully
    fetched. The 85% floor leaves room for the size constants above being approximate.
    """
    if not weights:
        return "ready"
    if all(w["bytes_present"] >= w["bytes_expected"] * 0.85 for w in weights):
        return "ready"
    if any(w["bytes_present"] > 0 for w in weights):
        return "partial"
    return "missing"


def _state(weights: list[dict[str, Any]], built: bool, setup_fetches: bool) -> str:
    """The backend's state, which is the build and the weights together.

    The subtlety is TRELLIS: its bootstrap installs the code and fetches nothing, so a
    built TRELLIS with no weights is *usable* -- the weights download on the first
    generation run. Reporting that as "missing" sent someone to a Set up button that
    re-ran a completed bootstrap.
    """
    if not built:
        return "missing"
    if not setup_fetches:
        return "ready"
    return _weights_state(weights)


def _action(weights: list[dict[str, Any]], built: bool, setup_fetches: bool) -> str:
    """What the button should offer: build, fetch, resume, or nothing."""
    if not built:
        return "build"
    if not setup_fetches:
        return "none"
    state = _weights_state(weights)
    return {"ready": "none", "partial": "resume"}.get(state, "download")


def _percent(present: int, expected: int) -> int:
    if expected <= 0:
        return 100
    return max(0, min(100, round(present / expected * 100)))


def _rank_key(backend: Backend) -> tuple[int, str]:
    return (backend.rank if backend.rank is not None else 99, backend.label)
