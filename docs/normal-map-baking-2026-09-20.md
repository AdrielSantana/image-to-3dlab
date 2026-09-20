# Normal-map baking: it works, and the thing that broke it was the sign

2026-09-20. Follows on from the retopology work in `journal/retopology-findings.md`, which
left normal baking as "the biggest unclaimed win, and the source is sitting unused in the
decode cache".

The source really was sitting there, and the bake really does work. Between those two
facts sat one defect that produced a spectacular-looking failure and four wrong
explanations.

## What the bake needs, and where it comes from

`generate.py` decimates the decoded mesh before it ever writes a GLB. For the Snag the
decoder produced **19,172,397 faces** and the shipped asset has 282,799 — 98.5% discarded.
But the decode is cached (`--dump-decode`, `scripts/patch_trellis_dump_decode.py`), so the
discarded detail is still on disk and costs nothing to recover.

`scripts/export_decode_highpoly.py` (new) turns that cache into the PLY the bake reads:

```bash
.venv/bin/python scripts/export_decode_highpoly.py run_decode.pt high.ply
.venv/bin/python scripts/blender_bake_normals.py low.glb high.ply normal.png --size 2048 --ray 0.004 --max-ray 0.004 --decimate-to 0
.venv/bin/python scripts/attach_normal_map.py low.glb normal.png out.glb --strength 0.6
```

Twelve minutes for the export, eight seconds for the bake, seconds for the attach.

## The failure: rainbow confetti

The first bake came back as a field of coloured cells with real relief visible underneath.
Measured rather than eyeballed: **48.1% of texels had a negative Z**, which a tangent-space
normal map cannot have — Z is the deviation *out of* the surface.

Four explanations were tested and all four were wrong. Recording them because each one
looked obvious at the time:

| Hypothesis | Test | Verdict |
|---|---|---|
| The bake source was shattered by decimation | Component count before/after | **Real, but not the cause.** See below |
| The PLY has no vertex normals, so it bakes flat facets | Smooth-shade the source, re-bake | No change (47.6%) |
| Ray distance jumps a gap to the wrong surface | Sweep 0.0003 → 0.015 | No change at any distance |
| Metal GPU bake bug | Re-bake on CPU | Identical to the texel |

Two controls proved the bake machinery itself was fine: a synthetic plane with sine bumps
baked correctly (3.6% negative Z), and baking the low-poly against *itself* gave flat
lavender (7.2%), as it must.

## The cause, measured inside Blender

A ray-cast probe replicating what the bake does — 2,000 rays from low-poly face centres,
recording distance and the angle between the low-poly normal and the hit normal:

```
samples 2000, misses 38, hits 1962
median hit distance   0.00131      (the surfaces are essentially coincident)
angle: p25 14.8°, median 72.4°, p75 164.2°
hits more than 90° from the low-poly normal: 48.9%
```

The rays land in the right place. They come back with the normal **reversed** — the upper
quartile sits at 164°, near-exactly opposite.

The reason is that the decode is non-manifold. The Snag arrives with 237,359 non-manifold
edges (measured during the retopology work), and winding cannot propagate across a
non-manifold edge, so no orientation-repair pass converges. `repair_decode.repair` still
earns its place — it fixed **87,198 inside-out components** on this mesh — but within the
one giant 19M-face component the winding stays mixed, and that is what the rays report.

## The fix: resolve the sign in the map

The mesh cannot be repaired by propagation, but the map does not need it to be. A
tangent-space normal has Z > 0 by construction, so a texel with Z < 0 can only have come
from a reversed source face, and the true normal is its negation.

`blender_bake_normals.py` now does this after every bake (`--keep-sign` opts out for
diagnosis) and prints the fraction it negated:

| | before | after |
|---|---|---|
| negative-Z texels | 48.1% | **0.1%** |
| mean blue channel | 132 | **221** |

The negated fraction is itself a useful read: near zero means a consistently wound source,
near half means a non-manifold one.

## The decimation trap, recorded separately

Before the sign was found, the first suspect was the bake source. That suspicion was
wrong about the confetti but right about the mesh:

| | components |
|---|---|
| Welded decode, 19,172,397 faces | **1,072** (99.7% of faces in one) |
| After `fast_simplification` to 1.5M | **215,842** — 20% of it in fragments under ten faces |

This is the same family as the 200k face-cap defect already documented in this repo:
collapsing edges on a mesh like this shatters it. `export_decode_highpoly.py` therefore
does **not** decimate by default, and warns loudly when asked to and the component count
explodes. Baking against the full 19.2M-face source takes eight seconds, so there is no
reason to reduce it.

## What it actually buys, on this asset

Modest. The Snag's retopologised mesh already follows the decode surface to a median
0.0013 (0.08% of the model's diagonal), so there is little lost meso-detail for a map to
restore — what comes back is bark and moss micro-relief, visible in raking light and in
the `0.6`-strength render, crunchy at `1.0`.

The lever should be larger wherever retopology genuinely smooths features away, which is
the case this was built for and has not been tested yet. `--strength 0.6` is the sane
default; `attach_normal_map.py` blends toward flat because trimesh drops glTF's
`normalTexture.scale` on export.

## Left open

- **A residual speckle in the map**, from rays hitting crevice walls at grazing angles.
  It reads as granularity rather than detail. A median filter or a lower strength both
  hide it; neither has been measured.
- **Ambient occlusion** from the same source, via `scripts/blender_bake_ao.py` — untried,
  and it will hit the same winding problem with no equivalent sign trick, so it likely
  needs the source oriented against the target first.
- **An asset where retopology loses real shape**, to measure what this is worth when there
  is something to recover.
