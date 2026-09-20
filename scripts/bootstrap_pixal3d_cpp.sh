#!/usr/bin/env bash
# Clone, build and weight the Pixal3D C++/GGML runtime on Apple Silicon.
#
#   scripts/bootstrap_pixal3d_cpp.sh
#
# Builds raven38/pixal3d.cpp into vendor/pixal3d-cpp and fetches the single-view Q8_0
# weight set (8.1 GB). Metal is automatic on Apple builds, so no backend flag is needed.
#
# This is the port to use on a Mac. The PyTorch one (pawel-mazurkiewicz/Pixal3D-mac) needs
# ~22 GB of weights resident and does not fit a 32 GB machine -- see
# docs/pixal3d-evaluation-2026-09-20.md.
#
# Needs the Metal compiler, which ships with full Xcode rather than the Command Line Tools.
# If `xcrun metal` fails, the two things that fix it are accepting the licence and
# downloading the toolchain component; both are printed below rather than run, because they
# need sudo or change a system-wide setting.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$REPO/vendor/pixal3d-cpp"
MODELS="$VENDOR/models/pixal3d-sv"
UPSTREAM="https://github.com/raven38/pixal3d.cpp.git"
WEIGHTS_REPO="raven38/pixal3d-sv-q8_0-v1"
MATTE_REPO="ilintar/trellis2-gguf"

say() { printf '\033[1;36m[pixal3d]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[pixal3d] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || die "Apple Silicon only."

for tool in cmake ninja git; do
    command -v "$tool" >/dev/null || die "$tool not found (brew install $tool)"
done

if ! xcrun --find metal >/dev/null 2>&1; then
    die "Metal compiler unavailable. With full Xcode installed, this is usually:
       sudo DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -license accept
       DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -downloadComponent MetalToolchain
     then re-run this script with DEVELOPER_DIR set."
fi

if [[ ! -d "$VENDOR/.git" ]]; then
    say "Cloning $UPSTREAM"
    git clone --depth 1 "$UPSTREAM" "$VENDOR"
fi

say "Fetching vendored ggml and friends"
git -C "$VENDOR" submodule update --init --recursive

say "Building (Metal is the default backend on Apple)"
cmake -S "$VENDOR" -B "$VENDOR/build" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$VENDOR/build" -j

[[ -x "$VENDOR/build/trellis-cli" ]] || die "build finished without trellis-cli"

# The weight set is self-contained -- four flow DiTs, three decoders, the image encoder and
# NAF -- but carries no matting model, so BiRefNet comes from the trellis2 GGUF repo. A
# pre-matted RGBA input skips matting entirely.
say "Fetching weights into $MODELS (8.1 GB, resumable)"
mkdir -p "$MODELS"
PYTHON="${PYTHON_BIN:-python3}"
HF_HUB_DISABLE_XET=1 "$PYTHON" - "$MODELS" "$WEIGHTS_REPO" "$MATTE_REPO" <<'PY'
import sys
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    sys.exit("huggingface_hub is not installed: pip install huggingface_hub")

models, weights_repo, matte_repo = sys.argv[1:4]
snapshot_download(weights_repo, local_dir=models, max_workers=2)
hf_hub_download(matte_repo, "q8/birefnet.gguf", local_dir=models)
print(f"weights in {models}")
PY

# trellis-cli looks for models flat in --models; the matte lands in a q8/ subdirectory.
if [[ -f "$MODELS/q8/birefnet.gguf" && ! -f "$MODELS/birefnet.gguf" ]]; then
    mv "$MODELS/q8/birefnet.gguf" "$MODELS/birefnet.gguf"
    rmdir "$MODELS/q8" 2>/dev/null || true
fi

say "Done. Generate with:"
echo "    python scripts/pixal3d_generate.py input.png output.glb --res 1024"
