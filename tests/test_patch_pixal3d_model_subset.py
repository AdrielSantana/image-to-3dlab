"""Tests for loading only the Pixal3D checkpoints a run needs.

The failure this addresses is not subtle — the process dies during model load on a 32 GB
machine, because 1024 wants ~22 GB of bf16 before sampling starts. What needs testing is
that the subtraction is exact (skipping the wrong model costs a whole run to discover) and
that the injected class-body code is valid Python, since it executes at import time.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_pixal3d_model_subset.py"


def _load():
    spec = importlib.util.spec_from_file_location("patch_pixal3d_model_subset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = _load()

ALL_MODELS = [
    "sparse_structure_flow_model",
    "sparse_structure_decoder",
    "shape_slat_flow_model_512",
    "shape_slat_flow_model_1024",
    "shape_slat_decoder",
    "tex_slat_flow_model_512",
    "tex_slat_flow_model_1024",
    "tex_slat_decoder",
]

ORIGINAL = '''class Pixal3DImageTo3DPipeline:
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(self):
        pass
'''


def test_nothing_is_dropped_when_the_variable_is_unset():
    assert patch.remaining_models(ALL_MODELS, "") == ALL_MODELS


def test_the_texture_stack_can_be_dropped_for_a_geometry_run():
    kept = patch.remaining_models(
        ALL_MODELS, "tex_slat_flow_model_512,tex_slat_flow_model_1024,tex_slat_decoder"
    )
    assert kept == [
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
    ]


def test_whitespace_and_empty_entries_are_tolerated():
    assert patch.remaining_models(ALL_MODELS, " tex_slat_decoder , ,") == [
        name for name in ALL_MODELS if name != "tex_slat_decoder"
    ]


def test_an_unknown_name_changes_nothing():
    """A typo must not silently drop a model that is needed."""
    assert patch.remaining_models(ALL_MODELS, "tex_slat_decoderr") == ALL_MODELS


def test_the_patch_is_idempotent():
    once = patch.apply(ORIGINAL)
    assert patch.is_patched(once)
    assert patch.apply(once) == once


def test_a_missing_anchor_fails_loudly():
    moved = ORIGINAL.replace("        'tex_slat_decoder',\n    ]\n", "    ]\n")
    with pytest.raises(SystemExit):
        patch.apply(moved)


def test_the_injected_class_body_runs_and_subtracts(monkeypatch):
    """It executes at import time, so a syntax or scoping slip breaks everything."""
    patched = patch.apply(ORIGINAL)
    monkeypatch.setenv(patch.ENV_VAR, "tex_slat_flow_model_1024,tex_slat_decoder")

    namespace: dict = {}
    exec(compile(patched, "patched_pipeline.py", "exec"), namespace)

    loaded = namespace["Pixal3DImageTo3DPipeline"].model_names_to_load
    assert "tex_slat_flow_model_1024" not in loaded
    assert "tex_slat_decoder" not in loaded
    assert "shape_slat_flow_model_1024" in loaded


def test_the_injected_body_leaves_the_list_alone_when_unset(monkeypatch):
    patched = patch.apply(ORIGINAL)
    monkeypatch.delenv(patch.ENV_VAR, raising=False)

    namespace: dict = {}
    exec(compile(patched, "patched_pipeline.py", "exec"), namespace)

    assert namespace["Pixal3DImageTo3DPipeline"].model_names_to_load == ALL_MODELS


def test_the_helper_names_do_not_leak_into_the_class(monkeypatch):
    """Class-body temporaries would otherwise become attributes on every pipeline."""
    patched = patch.apply(ORIGINAL)
    monkeypatch.setenv(patch.ENV_VAR, "tex_slat_decoder")

    namespace: dict = {}
    exec(compile(patched, "patched_pipeline.py", "exec"), namespace)

    cls = namespace["Pixal3DImageTo3DPipeline"]
    assert not hasattr(cls, "_i2l_os")
    assert not hasattr(cls, "_i2l_skip")
