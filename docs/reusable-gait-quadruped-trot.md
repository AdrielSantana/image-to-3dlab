# Reusable Rigify quadruped gait: accepted trot and implementation plan

Status: 2026-09-20. A repeatable, character-specific trot is working. Porting that
motion into the shared gait tool is the next step; portability has not yet been
demonstrated. Repairing the fox's low-head deformation is a separate task.

## Goal and supported target

Build a simple command-line tool that authors walk and trot actions on a rig
**generated from Rigify's `basic_quadruped` metarig**. The metarig defines the
skeleton; animation should drive the generated rig's controls.

The intended result is a useful starting animation across different body and
limb proportions, with explicit parameters for style. A matching rig does not
guarantee that a particular mesh or its weights can support every posture.
Only Blender 5.2.0 LTS was exercised in this session; broader version support
needs validation rather than inference from control names.

## What exists now

| Artifact | Status |
| --- | --- |
| `scripts/quadruped_gait.py` | Existing shared gait maths; predates the accepted trot below. |
| `scripts/blender_quadruped_walk.py` | Existing Blender driver; extend and validate it rather than adding a competing general-purpose entry point. |
| `tests/test_quadruped_gait.py` | Existing shared tests; not proof that the newer trot has been ported. |
| `characters/fox/fox_trot.py` | Repeatable recipe for the accepted fox trot. Its CLI runs `build()`, not the rejected neck experiments. |
| `characters/fox/test_fox_trot_blender.py` | Actual Blender rebuild comparison against the accepted action. |
| `output/monolith/moss-fox-trot-clean.blend` | Saved original fox skin and accepted `Fox_LowTrot_16f` action. |

`characters/` and `output/` are git-ignored local content, not dependencies a
fresh clone can access. Keep creature-specific calibration there. Only general
code belongs in `scripts/`, with entries in `scripts/README.md` and tests under
`tests/`. No new portable implementation is claimed by this document.

The [earlier investigation](quadruped-gait-2026-09-20.md) records the initial
walk/scamper attempts. Its preset values and conclusions are historical;
this document describes the later accepted trot and remaining work.

## The accepted motion

The user accepted the improved leg motion. The later attempt to flatten the
head/back silhouette was not accepted, and its mesh/weight changes were reverted.

The trot improved because of a combination of changes:

- Left front/right hind move together; the opposite diagonal is half a cycle later.
- A shorter cycle gives a quicker cadence without relying on exaggerated foot lift.
- Planted feet travel backward at one common speed in the in-place animation.
- Recovery includes paw curl and a smooth reversal behind and ahead of contact.
- The body compresses after contact, with modest separate chest and hip motion.
- The head and tail have restrained secondary motion.

These are the fox recipe's values, **not universal defaults**:

| Parameter | Accepted fox recipe |
| --- | --- |
| Period / frame rate | 16 frames at 24 fps |
| Loop endpoint | Frame 17 repeats frame 1; play frames 1–16 |
| Stance duty | 0.46 of a complete stride cycle |
| Stance fore/aft sweep | 0.40 scene units for every paw |
| Peak target clearance | 0.105 front / 0.115 hind scene units |
| Mean torso drop | 0.13 scene units |
| Torso bob amplitude | 0.022 scene units, two oscillations per cycle |
| Forward direction | World −Y on this asset |
| Corresponding forward travel speed | Approximately 1.30435 scene units/second |

The foot clearance and body values must become measured proportions or explicit
profile parameters in the shared tool. The crouch is a character choice, not a
requirement for all fox locomotion. This action is a diagonal trot; exact
direct-register footprint placement was not established by the tests.

## Mechanics to preserve in the shared implementation

### Common stance speed

Let `T` be cycle duration in seconds, `D` the fraction spent in stance, and `S`
the distance the paw sweeps backward during stance. The corresponding forward
speed is:

```text
speed = S / (D * T)
```

During stance, adding that forward character motion must leave the evaluated
paw stationary on the ground. Independently scaling front and hind sweeps by
their respective limb lengths generally produces different speeds and sliding.
Choose one travel speed, derive each stance sweep from its contact duration,
and check every limb's reach. Front/hind clearance can still differ.

The in-place test must be supplemented by a translating preview. A constant
paw height alone is not a test for sliding. Define scene units, rig scaling,
frame rate, and playback speed consistently in the speed report.

### Recovery and contact

The accepted recipe uses cubic Hermite fore/aft recovery with endpoint
velocities matched to stance, plus a `sin²(pi * swing_progress)` clearance arc.
Both position and velocity are continuous at lift-off and touchdown in the
underlying functions. Paw curl builds during recovery and returns to zero at
contact. Sampled linear keys approximate these curves; do not claim exact
subframe continuity from integer-frame tests alone.

The smooth turnaround extends slightly beyond the nominal stance endpoints.
Check the full evaluated trajectory when checking reach. Audit the paw sole,
not just the IK control origin: a rotated oversized paw can penetrate the floor
even when its target is above it.

### Body and legs remain coupled

Separate the implementation, but solve body motion and leg reach together.
Lowering the torso changes knee/elbow bend and available reach. Pose the body
first, evaluate its parent transforms, then place the IK targets to maintain
the intended paw contacts. A legs-only review mode should hold an explicit
body posture, not inherit arbitrary values from the previously active action.

Do not try to solve mesh folding by changing gait timing or foot trajectories.

## Three implementation responsibilities

| Part | Responsibilities | Must not contain |
| --- | --- | --- |
| Pure gait maths | Phases, duty, travel speed, foot paths, clearance, paw curl, body envelopes | `bpy`, asset names, fox-specific absolute measurements |
| Rigify adapter | Rig discovery, bone measurements, coordinate conversion, IK configuration, action creation, evaluated validation | Weight repair or assumptions that all bone rolls/object scales match the fox |
| Optional posture profile | Body height/compression, chest and hip rotation, head/tail motion, explicit reference alignment | Hidden dependence on the active pose or automatic mesh edits |

Begin with walk and trot. Preserve the existing `scamper` terminology as a
style/preset where appropriate; do not assume that faster cadence alone creates
a different biological gait. A lope/gallop needs its own contact pattern and
validation.

## Rigify adapter requirements

- Identify the generated rig explicitly; reject ambiguity between multiple rigs
  and distinguish it from the metarig. Validate the required control contract.
- Derive forward/lateral/up axes and lengths from rest bones in a consistent
  coordinate space. A posed mesh bounding box is not a stable size measure.
- Support rotated and uniformly scaled rigs deliberately. Detect and reject
  nonuniform/mirrored transforms initially unless the implementation tests them.
- Preserve Rigify's internal `MCH-`, `ORG-`, and `DEF-` state. Resetting all pose
  bones previously broke the leg mechanism before any animation was authored.
- Inspect the IK/FK, stretch, parenting and pole settings. Key the required
  state into the new action so it does not depend on an earlier action.
- Place targets through evaluated parent transforms. Rest-bone local offsets
  alone are insufficient for controls with rotated mechanism parents.
- Preserve current actions; refuse duplicate output action names by default.
  Validate active NLA blending before authoring and avoid silently muting it.
- Freeze or explicitly load reference alignment. Re-reading the active head
  pose on every build compounds correction offsets.
- When editing copied curves, key the changed value before an evaluation can
  restore the old curve value. Test the evaluated result afterward.
- Include the closing key but exclude it from the looping playback range.
- Validate before saving a final output. Report unsupported rigs and unreachable
  motion; do not silently overwrite a good file with a failed candidate.

## What went wrong with the fox's low head

Much of the skull and ears followed `DEF-spine.010`, while a comparatively small
part of the muzzle followed `DEF-spine.011`. Rotating the neck substantially
while keeping the head level distorted the face. Moving the complete head/neck
chain kept the face closer to its shape but compressed the chest. Subsequent
weight/rebind experiments produced a ledge and accordion-like folds below the
neck. The user rejected those results.

Those experiments could preserve the reference pose numerically while still
looking wrong in the bent pose. A passing bind-point or loop test does not
validate skin deformation. All experimental skin changes were reverted, and
the accepted trot uses the original mesh/weights. The retained local
`fox_skin.py` and `lower_head()` experiments are **not accepted fixes**.

The evidence establishes a deformation problem with the requested posture; it
does not establish that the entire model must be replaced. A separate repair
should inspect the rest neck, layered throat-fur geometry, bone placement and
weight transitions. Test neutral, lowered and extended head poses first, then
reapply the unchanged trot. Consider local reshaping/rebinding or a corrective
shape only after identifying which part causes the fold.

## Verification so far and the portability gate

The local tests passed: 13 unit tests, with the Blender integration test run
separately. Two independent rebuilds matched the accepted action over all 46
deform bones and 17 frames, with maximum matrix-component difference about
`1.32e-6`. The second rebuild began with a rebuilt action active, checking that
reference corrections did not accumulate. The mesh was unchanged.

That establishes repeatability on this fox, not portability. Before advertising
the shared generator, require:

1. Unit tests on the actual imported maths: contact sequences, common ground
   speed, phase boundaries, periodicity, parameter validation and scaling.
2. Blender integration tests using a stock generated basic quadruped and rigs
   with different limb/body proportions, rotation and uniform scale. Fixtures
   should be generatable without git-ignored creature assets.
3. Evaluated checks for target accuracy, reachable limb poses, unintended
   stretching, loop closure and state independence across repeated builds.
4. Grounded translating previews and visual checks of mesh soles, knees/hocks,
   paw recovery and body motion on at least two skinned characters.
5. A fox regression comparison using its local profile. The existing shared
   scripts must not be described as reproducing this trot until they do.

## Work split and handoff

The user chose to **defer the shared-script implementation to another agent**
and continue the fox model repair separately. The saved fox recipe and clean
blend are the reference; do not overwrite either during either workstream.

Implementation checklist for the shared-tool agent:

- [ ] Read the existing shared scripts and tests before editing; preserve concurrent work.
- [ ] Extract the accepted common-speed foot paths and recovery curl into pure functions.
- [ ] Replace asset-specific numbers/alignment with measured proportions and explicit profiles.
- [ ] Update the existing Blender adapter with the preflight and action-preservation contract above.
- [ ] Add generated-rig integration fixtures and the portability checks above.
- [ ] Reproduce the accepted fox trot from its local profile without changing its mesh.
- [ ] Update the script registry, usage documentation and changelog with verified support limits.

Do not port the rejected `fox_skin.py` or `lower_head()` experiments into the
shared generator. The model-repair work is independent of this checklist.

Within the shared-tool work, first port the accepted foot mechanics into `quadruped_gait.py`, add an explicit
measured/profile-driven adapter in `blender_quadruped_walk.py`, and verify it on
simple rigs with conservative posture. Preserve the accepted fox recipe as the
regression reference. Only then add optional posture controls and verify the
fox profile without forcing the rejected low-head pose.

This advances the reusable-tool goal without treating one creature's skin
defect as a gait requirement. If another mesh deforms correctly with the same
motion while the fox does not, that is useful evidence for the separate repair.
Every shipped script must generalize, be indexed in `scripts/README.md`, and
have tests. Per-character recipes, profiles and model repairs stay local.

## References

- [Kate MacQuarrie: canine gaits](https://www.pei-untamed.com/post/canine-gaits)
  — observed fox travel patterns and faster trot/lope transitions.
- [Jim Bradrick: creating a four-legged cycle](https://acmepunched.blogspot.com/2014/06/no-66-creating-four-legged-walk-cycle.html)
  — an animator's fox-trot construction and body/paw treatment.

These informed the motion direction. The recipe values above are animation
choices measured on the local character, not measurements from those references.
