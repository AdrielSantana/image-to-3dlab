# Quadruped gait from first principles — moss fox, 2026-09-20

> Historical investigation: this predates the later accepted 16-frame trot and
> the rejected low-head deformation experiments. See
> [Reusable Rigify quadruped gait](reusable-gait-quadruped-trot.md) for the current
> status, verification limits and implementation plan.

Building a reusable walk/trot for the Rigify `basic_quadruped` metarig, tested on the
moss fox. Everything here is measured on this repo's own assets; the numbers are the
point, not the prose.

## What shipped

| File | What it is |
|------|------------|
| `scripts/quadruped_gait.py` | Pure gait maths — footfall phases, duty, foot arcs, body bob, tail/head secondary motion, and the self-audit. No `bpy`, 46 tests. |
| `scripts/blender_quadruped_walk.py` | The Blender layer: measures the rig, calibrates axes, keys, audits. |

Everything is a multiple of **the rig's own proportions**, never an absolute copied
between creatures. The ram and the fox are within 5% in height but have *inverted* limb
segments (ram's shin is its longest hind bone, fox's metatarsus is), so shared absolutes
over-reach the IK targets and skate the feet.

## The five bugs, in the order they were found

Each was invisible in the report of the run that produced it — the action looked
authored, the curves were there, and the result was wrong.

1. **Bone heat failed on every bone at once.** The voxel proxy was watertight (0 boundary
   edges) but carried 13 orphan 8-vertex specks. Bone heat solves one global linear
   system, so an island with no bone in it makes it singular: *all* 34 groups came back
   empty. Stripping loose islands took it to 34/34 with 0 unweighted vertices.

2. **Object scale silently coarsened the proxy.** `voxel_size` is local space,
   `mesh.dimensions` is world. A mesh at scale 1.5 got a proxy 1.5x coarser than asked —
   enough to fuse a quadruped's legs.

3. **A blanket pose reset broke the rig.** Zeroing *every* pose bone included Rigify's
   `MCH-` machinery. This quadruped keeps ~65° on `MCH-front_foot_parent.L/R` that the
   foreleg IK depends on; flattening it hyperextended the elbow to 180° and folded the
   wrist to 118° **before a single keyframe existed**. It presents as a bad walk, which
   is where three rounds of fixing the animation went. Reset controls only.

4. **The march.** Foot lift was set independently of stride, then scaled 2.4x while
   stride stayed put — so the foot rose as far as it travelled. That is a parade step by
   definition. Lift is now *derived* from stride.

5. **The disco strut.** Lateral body roll equal to the vertical bob, at stride frequency,
   with the tail swinging at that same frequency — the two reinforce. Kept as
   `QuadWalk_BeeGees` for reference. Roll is now 0.35x the bob.

## The invariants worth keeping

Measured independently on two animals of different proportions:

- **lift / stride ≈ 0.27** (ram's accepted walk: 0.268 hind, 0.275 front). Break it and
  the gait marches. Our worst was 1.18.
- **body bob ≈ 0.77% of height** for a walk.
- **stride ≈ 0.25x leg length** for a walk.
- Duty 0.65 walk, 0.50 trot, 0.45 scamper.

## The constraint specific to this fox

Its **foreleg rests at 94.3% of full IK extension** — ±0.016 of reach each way, against a
requested ±0.077. The hind leg rests at 83.5% with ±0.12 and was never affected, which is
why the defect looked like a front-leg-only mystery for hours.

Consequences:

- Any meaningful stride hyperextends the elbow to a locked 180°, and the leftover bend
  lands in the wrist — reads as *the ankle shoved out in front of the leg*.
- Lowering the body folds the limb and buys reach: a 0.08 drop takes available stride
  from 0.032 to 0.371. **Create the headroom and then spend it**, or a bigger lift just
  marches.
- This is an artefact of the generated rest pose, not of Rigify. A source image with a
  more naturally bent foreleg would avoid it.

## Things measured and deliberately *not* acted on

- **Foreleg rake is reported but not gated.** The metric gave 6.2° off the authoring
  bones and 10.6° off the evaluated depsgraph for the identical cycle, and its sign
  disagrees with the renders. A gate on it would fail correct cycles. Judge by eye until
  the measurement is validated. `GAIT_LIMITS["front_rake_excursion_deg"] is None`.
- **Shoulder rotation defaults to 0.** Intended to absorb foreleg reach; measured, it
  hyperextends the elbow to 180° on its own. Left as an opt-in knob.
- **Body drop defaults to 0** for the same reason, though the original evidence against
  it was gathered on the rig damaged by bug 3 and is invalid. It behaves correctly now.

## Head yaw: read the failure carefully

The fox's head sits ~33° to its own left — measured on the mesh (head vs neck vertex
centroids), and inherited from the source image, not a rig error. Correcting it all on
the `head` bone **squashed the muzzle**: only 389 head vertices are rigidly weighted while
**9,704 (96%) sit in the blend zone** with the neck, so a large rotation stretches rather
than moves them. Splitting the correction across neck and head (43/57) fixes it. The real
lever is weight tightness — those soft weights come from the voxel proxy, so a finer
`--voxel-fraction` would give the head a rigid core.

There are **no ear, jaw or face vertex groups at all**, so those cannot be animated.

## Open

- It still reads as a purposeful walk, not a fox trot: the body does not drop per stride
  and the limbs do not tuck enough. GPT Astra is taking a pass.
- No forward root motion — every cycle is in place, which flatters a march.
- Foxes travel by **direct-register trot** (hind foot lands in the front foot's print),
  not by walking. Not yet implemented.
