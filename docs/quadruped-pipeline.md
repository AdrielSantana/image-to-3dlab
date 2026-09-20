# Fitted quadruped → weights → reference pose → locomotion

Use `scripts/blender_quadruped_pipeline.py` for a staged, inspectable Rigify workflow.
It packages the process developed during the moss-fox session, **not that asset's
hard-coded proportions or pose**. A person fits the metarig; a person or LLM tunes
the character profile and reviews deformation. The tool handles repeatable mechanics.

Supported starting contract: a fitted Rigify **Basic Quadruped** metarig, then its
generated rig with IK paws. Tested with Blender **5.2.0 LTS**. The action API uses
slots/channel bags; older Blender releases are not certified. Keep Blender/Rigify
versions fixed between runs. No network service, API key or LLM is required.

## 1. Fit and bind

Import your character and fit the metarig **in Edit Mode**, including front/top
views. Place the joints inside the underlying anatomy, not on decorative fur.
Save `fitted.blend`. Replace the example object names with yours:

```sh
blender --background fitted.blend --python-exit-code 1 \
  --python scripts/blender_quadruped_pipeline.py -- bind \
  --mesh Creature --metarig metarig \
  --voxel-fraction 0.006 --out bound.blend
```

The script:

1. Rejects already-bound meshes, shape keys/modifiers, posed metarigs, existing
   Rigify targets, and nonuniform/reflected object transforms.
2. Bakes metarig object transforms **without moving fitted world-space joints**.
3. Generates Rigify and checks both endpoints of each original bone against the
   fitted metarig (allowing Rigify to reverse its tail chain).
4. Voxel-remeshes a disposable copy, keeps its largest connected component, runs
   bone heat there, and transfers the resulting weights onto the original surface.
5. Normalizes weights and checks coverage, geometry and UV preservation.

**The visible character is not remeshed.** Proxy island removal affects only the
disposable copy, not the original leaves, ears, flowers or other details.

`--voxel-fraction` is a fraction of the mesh's longest world-space dimension.
Smaller resolves narrower gaps but can preserve problematic tiny islands; larger
can fuse legs or lose narrow features. Try one change per new output. A heat solver
failure is an explicit failure, not a reason to accept empty weights.

Unused deform bones fail by default. If an unused bone is genuinely intentional,
repeat `--allow-unused BONE_NAME` for those specific bones **after inspecting them**.
Do not whitelist a whole side of an animal to make a broken bind pass.

Open `bound.blend` and inspect small head turns, head lowering, crouching and one
paw lift at a time. Vertex coverage does **not** prove correct anatomical weights.
The proxy can bridge adjacent limbs; nearest-surface transfer can put decorations
on the wrong region. Fix fitting/weights or perform local mesh cleanup as needed.

## 2. Pose and capture

Pose the generated rig into the desired average locomotion posture. Leave paws in
IK (`IK_FK=0`), keep their soles at your intended ground level, and mute NLA tracks.
Save `posed.blend`. Then capture that exact state:

```sh
blender --background posed.blend --python-exit-code 1 \
  --python scripts/blender_quadruped_pipeline.py -- capture \
  --rig rig --out posture.json
```

Add `--frame 12` to capture a specific frame of an existing action. Without it,
the saved current frame/manual pose is used. The JSON records control basis
matrices, rotation modes, numeric custom properties, IK target matrices, rest
skeleton signature, travel axes and support height. It is not a universal pose
that can be applied to a different skeleton: **capture one per fitted character**.
Recapture after changing rest bones or the rig's object rotation/scale.

Standard Rigify names are a rig interface, not character names. For renamed
controls or a different tail chain, pass `--controls controls.json`, e.g.:

```json
{
  "controls": {"head": "head_control", "FL": "front_paw_target.L"},
  "tail": ["tail_01", "tail_02", "tail_03"]
}
```

Omitted mappings retain standard names. Valid mapping keys are `FL`, `FR`, `BL`,
`BR`, `torso`, `chest`, `hips`, `neck`, `head`. An empty tail list disables tail
animation. This mapping is **not** an adapter for arbitrary rig mechanisms.

## 3. Animate and tune

Create a small `motion.json` (omitted settings use defaults):

```json
{
  "gait": "trot",
  "frames": 16,
  "fps": 24,
  "duty": 0.46,
  "stride": 0.35,
  "lift": 0.10,
  "curl_degrees": 16,
  "bob": 0.02,
  "chest_motion": 1,
  "hip_motion": 1,
  "neck_motion": 1,
  "head_compensation": 1,
  "body_roll_degrees": 0.35,
  "tail_pitch_degrees": 0.65,
  "tail_yaw_degrees": 0.9,
  "tail_lag": 0.45
}
```

```sh
blender --background posed.blend --python-exit-code 1 \
  --python scripts/blender_quadruped_pipeline.py -- animate \
  --rig rig --pose posture.json --settings motion.json \
  --name TravelTrot --transition --out animated.blend
```

Outputs: `TravelTrot_Reference`, `TravelTrot`, and, with `--transition`,
`TravelTrot_Transition`. Existing actions remain intact; collisions are rejected.
The transition is active when requested. All authored clips are **in place**;
no root displacement or engine export is performed.

| Knob | Meaning / first adjustment |
|---|---|
| `gait` | `walk` uses staggered four-foot phases; `trot` uses diagonal pairs. A scurry is a tuned trot, not a separate gait model. |
| `frames`, `fps` | Cycle duration. Defaults: trot 16, walk 28, at 24 fps. |
| `duty` | Fraction of cycle spent in stance. Defaults: trot .46, walk .65. |
| `stride` | Fore/aft travel as a fraction of rest support height. Reduce if limbs overextend. |
| `lift` | Swing clearance as a fraction of support height. Reduce a marching/high-stepping look. |
| `curl_degrees` | Swing paw rotation; reduce if toes scrape or curl the wrong way for a custom rig. |
| `bob` | Torso vertical amplitude as a fraction of support height. |
| `chest_motion`, `hip_motion`, `neck_motion` | Multipliers for small phase-staggered regional bends and translations. Zero disables that region's added motion. |
| `head_compensation` | Small compensating nod; intended to work with the neck, not replace it. |
| `body_roll_degrees` | Small lateral body rotation per stride. |
| `tail_pitch_degrees`, `tail_yaw_degrees` | Rotation amplitude **per tail segment**, so longer chains may need smaller values. |
| `tail_lag` | Phase lag in radians per tail segment. |
| `transition_hold` | Frames to hold at each end; default 12. |
| `transition_blend` | Frames for each lowering/start/stop/rising stage; default 16. |
| `transition_cycles` | Full-speed cycles between start and stop; default 3. |

Support height is derived from rest chest/hips above the mean rest IK-foot height,
not from the tail-inclusive bounding box. The generator restores the frozen
reference each frame, modifies control bones only, then positions IK feet after
body motion. It never resets Rigify's internal mechanism bones. The repeated
endpoint is keyed but excluded from loop playback.

The transition starts/ends in zero-transform **standing**, not a lying/sitting rest
pose or an idle animation. It eases into the reference, fades locomotion in/out,
and returns to standing. It is a preview blend, not a physically simulated braking
or acceleration controller. Inspect foot sliding during those transition sections.

## Reports, safety and determinism

Each stage writes `OUT.report.json` (override with `--report`). Reports include
input blend SHA-256, Blender version, arguments, and stage-specific checks;
animation also records resolved settings and pose hash. Failed processing writes
a failure report and raises: use `--python-exit-code 1` so a shell sees failure.
The output blend is saved only after validation; input files, existing outputs,
reports and action names are never silently overwritten. Run on copies in a
background process: an interactive failure can still leave partial in-memory work.

Repeatability means the same saved input/profile/settings and pinned software
produce numerically equivalent poses. It does **not** promise byte-identical blend
files, cross-version bone-heat results, or good automatic weights on every mesh.
Pin this repository revision too. Source asset licensing/provenance still applies;
these Blender processing reports do not replace generation provenance sidecars.

Validation currently checks generated joint alignment, weight coverage and sums,
visible mesh/UV preservation, finite pose matrices and loop closure. It does not
yet automatically certify sole penetration, reach margins, skin self-intersection,
foot sliding, or aesthetic quality. Those are required visual review gates.
Walk is a starting preset, not an approved walk for every creature. Attack, death,
jump and pain reactions are **not implemented** by this script.

## Working with an LLM in your own session

Give the agent this document, your saved input, report and a short viewport clip.
Ask it to:

1. Inspect the report and fitting before editing animation. Missing side-specific
   weights suggest alignment/proxy problems, not a need for larger foot motion.
2. Preserve a reference blend/profile and choose new output/action names per test.
3. Change one parameter group at a time: posture → feet/timing → body/neck → tail.
4. Compare opening frames and a full cycle. Test a crouch and head turn separately
   if deformation fails; don't hide bad weights with new animation offsets.
5. Record accepted settings and why, then rerun the same inputs to verify recovery.

Example request: “Use my fitted metarig and this pipeline. Don't remesh the visible
mesh. Check binding first; capture this crouch; make a short diagonal trot. If it
marches, inspect stride, clearance and cadence. Keep reports and earlier versions.”

## Tests

```sh
.venv/bin/python -m pytest tests/test_blender_quadruped_pipeline.py tests/test_scripts_registry.py -q
blender --background --factory-startup --python-exit-code 1 \
  --python tests/test_quadruped_pipeline_blender.py
```

Pure tests exercise settings, phases, stance speed, clearance, transition timing,
alignment and overwrite protection. Blender fixtures exercise stock Rigify,
transformed rigs, reference recovery and repeatable action generation. A real
character bind was also checked during development. A second independently
skinned character still needs visual acceptance before broader quality claims.
