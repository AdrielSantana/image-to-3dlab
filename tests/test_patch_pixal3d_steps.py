"""Tests for letting Pixal3D run fewer sampling steps.

What matters: a bad `PIXAL3D_STEPS` never silently changes a run, the patch lands on the
one function every stage samples through, it is idempotent, and it refuses rather than
guesses when upstream moves the anchor.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "patch_pixal3d_steps.py"
VENDOR_SOURCE = REPO / "vendor" / "pixal3d-cpp" / "src" / "flow_runner.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("patch_pixal3d_steps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = _load()

# The shape of upstream's sample_flow as of raven38/pixal3d.cpp 2026-09-23.
ORIGINAL = '''#include <cstdlib>
#include <cstdio>

std::vector<float> sample_flow(const FlowFwdProj& fwd, std::vector<float> sample,
                               const float* cond, const float* neg_cond,
                               const float* proj, const float* neg_proj,
                               const SamplerParams& sp,
                               std::vector<std::vector<float>>* trace) {
    const float sm = sp.sigma_min;
    std::vector<float> ts(sp.steps + 1);
}
'''


@pytest.mark.parametrize("raw, expected", [
    ("8", 8), ("1", 1), ("50", 50), (" 8", 8), ("+8", 8),
])
def test_whole_numbers_in_range_set_the_steps(raw, expected):
    assert patch.steps_override(12, raw) == expected


@pytest.mark.parametrize("raw", [None, "", "0", "51", "-3", "abc", "8.5", "8 ", "1_0"])
def test_anything_else_keeps_the_default(raw):
    """`PIXAL3D_STEPS=0` or a typo must not quietly change what the run does."""
    assert patch.steps_override(12, raw) == 12


def test_the_patch_copies_the_params_and_announces_the_override():
    patched = patch.apply(ORIGINAL)
    assert patch.is_patched(patched)
    assert "const SamplerParams& sp_in," in patched
    assert "SamplerParams sp = sp_in;" in patched
    assert 'std::getenv("PIXAL3D_STEPS")' in patched
    assert "PIXAL3D_STEPS=%d overrides %d steps" in patched
    # The helper is defined before the function that calls it.
    assert patched.index("static int i2l_steps_override") < patched.index("sample_flow(")
    # The rest of the sampler still reads `sp`, now the local copy.
    assert "std::vector<float> ts(sp.steps + 1);" in patched


def test_the_patch_is_idempotent():
    once = patch.apply(ORIGINAL)
    assert patch.apply(once) == once
    assert once.count("i2l_steps_override(") == 2  # definition + one call


def test_a_moved_anchor_is_refused_not_guessed():
    with pytest.raises(SystemExit, match="anchor not found"):
        patch.apply(ORIGINAL.replace("const SamplerParams& sp,", "SamplerParams sp,"))


def test_missing_includes_are_refused():
    with pytest.raises(SystemExit, match="lacks"):
        patch.apply(ORIGINAL.replace("#include <cstdlib>\n", ""))


@pytest.mark.skipif(not VENDOR_SOURCE.is_file(), reason="pixal3d.cpp is not checked out")
def test_the_real_flow_runner_accepts_the_patch():
    """Against the shipped source, not a copy: the anchors must be present (or it is
    already patched)."""
    source = VENDOR_SOURCE.read_text()
    patched = patch.apply(source)
    assert patch.is_patched(patched)
    assert patched.count("i2l_steps_override(") == 2
