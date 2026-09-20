"""Tests for exposing Pixal3D's low-VRAM hooks.

The hooks exist upstream and are unreachable: nothing sets `args.low_vram`. What matters
here is that the environment variable reads the way a person expects — `=0` must not turn
it on — and that the injected condition is valid Python using an import the host file
actually has.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_pixal3d_low_vram.py"


def _load():
    spec = importlib.util.spec_from_file_location("patch_pixal3d_low_vram", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = _load()

ORIGINAL = '''import os
import tempfile


def image_to_asset(pipeline, image, args, device=None, tmp_dir=None):
    if getattr(args, "low_vram", False):
        _install_low_vram_hooks(pipeline, device)
    return "asset"
'''


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_truthy_values_enable_it(value):
    assert patch.low_vram_requested(False, value) is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "False", "   "])
def test_falsy_values_leave_it_off(value):
    """`PIXAL3D_LOW_VRAM=0` in a profile must not silently enable it."""
    assert patch.low_vram_requested(False, value) is False


def test_the_existing_args_path_still_wins():
    assert patch.low_vram_requested(True, None) is True
    assert patch.low_vram_requested(True, "0") is True


def test_applying_the_patch_keeps_the_args_path():
    patched = patch.apply(ORIGINAL)
    assert patch.is_patched(patched)
    assert 'getattr(args, "low_vram", False)' in patched
    assert "PIXAL3D_LOW_VRAM" in patched


def test_the_patch_is_idempotent():
    once = patch.apply(ORIGINAL)
    assert patch.apply(once) == once


def test_a_missing_anchor_fails_loudly():
    moved = ORIGINAL.replace(
        '    if getattr(args, "low_vram", False):\n', "    if args.low_vram:\n"
    )
    with pytest.raises(SystemExit):
        patch.apply(moved)


def test_the_host_file_must_import_os():
    """Injected code may not assume the host's imports."""
    assert patch.uses_os_module(ORIGINAL) is True
    assert patch.uses_os_module(ORIGINAL.replace("import os\n", "", 1)) is False


def test_the_patched_condition_compiles_and_branches(monkeypatch):
    patched = patch.apply(ORIGINAL)
    namespace: dict = {"_install_low_vram_hooks": lambda *a: namespace.__setitem__("hooked", True)}
    exec(compile(patched, "patched_generate_mps.py", "exec"), namespace)

    class Args:
        low_vram = False

    monkeypatch.setenv("PIXAL3D_LOW_VRAM", "1")
    namespace["image_to_asset"](None, None, Args())
    assert namespace.get("hooked") is True

    namespace.pop("hooked", None)
    monkeypatch.setenv("PIXAL3D_LOW_VRAM", "0")
    namespace["image_to_asset"](None, None, Args())
    assert namespace.get("hooked") is None
