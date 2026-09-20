"""Tests for the Pixal3D BRIA substitution.

This is a licence guardrail, not a preference, so the properties that matter are that it
fires on the exact string the shipped weights ask for, that it cannot be applied twice,
and that it fails loudly when the anchor moves — a silently skipped patch here means a run
downloads BRIA RMBG-2.0.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_pixal3d_rembg.py"


def _load():
    spec = importlib.util.spec_from_file_location("patch_pixal3d_rembg", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = _load()

# The upstream file as it ships, trimmed to what the patch touches.
ORIGINAL = '''from typing import *
from transformers import AutoModelForImageSegmentation


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
'''


def test_the_exact_name_the_shipped_weights_ask_for_is_substituted():
    """pipeline.json in TencentARC/Pixal3D carries this literal string."""
    assert patch.permissive_model_name("briaai/RMBG-2.0") == "ZhengPeng7/BiRefNet"


def test_any_bria_model_is_substituted_not_just_that_one():
    assert patch.permissive_model_name("briaai/RMBG-1.4") == "ZhengPeng7/BiRefNet"


def test_a_permissive_model_is_passed_through_untouched():
    assert patch.permissive_model_name("ZhengPeng7/BiRefNet") == "ZhengPeng7/BiRefNet"
    assert patch.permissive_model_name("some/other-matting") == "some/other-matting"


def test_applying_the_patch_inserts_the_guard():
    patched = patch.apply(ORIGINAL)
    assert patch.is_patched(patched)
    assert "briaai/" in patched
    assert "ZhengPeng7/BiRefNet" in patched
    # The guard must sit inside __init__, before the model is fetched.
    assert patched.index(patch.MARKER) < patched.index("from_pretrained")


def test_the_patch_is_idempotent():
    once = patch.apply(ORIGINAL)
    twice = patch.apply(once)
    assert once == twice
    assert once.count(patch.MARKER) == twice.count(patch.MARKER)


def test_a_missing_anchor_fails_loudly():
    """Silently doing nothing here would mean a run loads BRIA."""
    moved = ORIGINAL.replace(
        'def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):',
        "def __init__(self, model_name=None, **kwargs):",
    )
    with pytest.raises(SystemExit):
        patch.apply(moved)


def test_the_injected_guard_is_valid_python_and_behaves():
    """Compile the patched file and drive the guard, rather than trusting the string."""
    patched = patch.apply(ORIGINAL)
    # Keep only the injected branch, with the model load stubbed out.
    body = patched.replace(
        """        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
""",
        "        self.model_name = model_name\n",
    ).replace("from transformers import AutoModelForImageSegmentation\n", "")

    namespace: dict = {}
    exec(compile(body, "patched_BiRefNet.py", "exec"), namespace)

    assert namespace["BiRefNet"]("briaai/RMBG-2.0").model_name == "ZhengPeng7/BiRefNet"
    assert namespace["BiRefNet"]("ZhengPeng7/BiRefNet").model_name == "ZhengPeng7/BiRefNet"
