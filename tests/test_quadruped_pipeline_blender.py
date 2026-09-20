"""Blender integration: run with Blender --background --python this file.

Ordinary pytest skips these tests; no external character assets are needed.
"""

import sys
from pathlib import Path

if __name__ != "__main__":
    import pytest

    pytest.importorskip("bpy")
    pytest.importorskip("mathutils", reason="Requires real Blender, not a bpy test stub")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import blender_quadruped_pipeline as pipeline


def test_stock_rig_and_rotated_scaled_variant():
    import bpy
    from mathutils import Matrix

    bpy.ops.preferences.addon_enable(module="rigify")
    bpy.ops.object.armature_basic_quadruped_metarig_add()
    meta = bpy.context.object
    bpy.ops.pose.rigify_generate()
    rig = meta.data.rigify_target_rig
    for variant in range(2):
        pipeline.activate(rig, None)
        for p in pipeline.controls(rig):
            p.matrix_basis = Matrix.Identity(4)
        if variant:
            rig.rotation_euler.z = 0.7
            rig.location = (2, -3, 0.4)
            rig.scale = (1.7,) * 3
        bpy.context.view_layer.update()
        pipeline.require_rig(bpy, rig.name)
        profile = pipeline.capture(
            bpy, rig, pipeline.DEFAULT_CONTROLS, pipeline.DEFAULT_TAIL
        )
        cfg = pipeline.settings({"transition_cycles": 1})
        result = pipeline.animate(
            bpy, rig, profile, cfg, f"Fixture_{variant}", transition=True
        )
        assert result["loop_error"] < 1e-5
        # Same frozen profile twice must reproduce evaluated matrices.
        source = bpy.data.actions[f"Fixture_{variant}"]
        pipeline.activate(rig, source)
        expected = pipeline.evaluated_matrices(bpy, rig, 5)
        pipeline.animate(bpy, rig, profile, cfg, f"Repeat_{variant}")
        actual = pipeline.evaluated_matrices(bpy, rig, 5)
        error = max(
            abs(expected[n][i][j] - actual[n][i][j])
            for n in expected
            for i in range(4)
            for j in range(4)
        )
        assert error < 1e-5, error
        # Reference action recovers exact captured control transforms.
        pipeline.activate(rig, bpy.data.actions[f"Repeat_{variant}_Reference"])
        pipeline.evaluated_matrices(bpy, rig, 1)
        for n, state in profile["bones"].items():
            assert (
                max(
                    abs(rig.pose.bones[n].matrix_basis[i][j] - state["basis"][i][j])
                    for i in range(4)
                    for j in range(4)
                )
                < 1e-5
            )
    print(
        "QUADRUPED_PIPELINE_BLENDER_PASS: stock + rotated/uniform-scaled; repeatability; reference; transition"
    )


if __name__ == "__main__":
    test_stock_rig_and_rotated_scaled_variant()
