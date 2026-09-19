# Repository guidance

## Blender animation work

Before writing new Blender animation automation, inspect the existing
`scripts/blender_*.py` tools, especially `scripts/blender_inspect.py` for the
shared local RPC client.

Per-creature rigs, animations and their tests live in `characters/<name>/`, which
is **git-ignored** — they are our own content, not part of the pipeline a user
clones. Each folder keeps its own recipe index (`SNAG.md`, `PANGOLIN.md`,
`FOREST_FLICKER.md`). If you do not have that folder, you do not need it.

Those recipes are a **model-specific library**, not a general rig-independent
toolkit. Reuse/adapt proven techniques; do not execute a recipe on a
different model merely because it has similarly named bones. Prefer extracting
shared operations with explicit rig/action/path inputs when the task calls for
generalization. Do not create another near-duplicate without checking this library.

Never import recipe modules for discovery: many edit the live scene and save at
module scope. Read source, verify the current file/rig/actions and output targets,
and preserve unsaved work and original actions before authorized edits. Superseded
experiments are documented in the creature indexes and are not approved fixes.

Check animation visually from relevant angles and intermediate frames, alongside
timing, deformation, ground contact and action-switch checks. Syntax checks are
not animation validation. Saved `.blend` assets are external to this repository;
committing recipes does not back them up.
