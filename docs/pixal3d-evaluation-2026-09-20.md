# Pixal3D beats our TRELLIS.2 route, and the C++ port is why it runs at all

2026-09-20, evening. Verdict from the user, on the moss fox: **"Pixal3D is the best — by a
mile."** This is the write-up of how that was reached, including the hour spent on the
wrong Mac port first.

## The result

| | faces | wall clock | size | colour |
|---|---|---|---|---|
| TRELLIS.2 1024_cascade | 289,566 | 14.3 min | 13 MB | **bleached** — cream and pale yellow where the source is green |
| TRELLIS.2 512 | 290k | — | 13 MB | muted but closer |
| TRELLIS.2 → retopo → Hunyuan 2.1 PBR repaint | 39,962 | 14.3 min + ~2 + 348s | 4.4 MB | saturated, runs warm |
| **Pixal3D SV q8_0, one pass** | 931,268 | **5 min 50** | 36.9 MB | **saturated straight out** |
| Pixal3D + retopo + compress | 39,987 | +7 s | **3.2 MB** | holds |

Comparison renders: `output/pixal3d-test/fox_four_way.jpg` and
`fox_finished_comparison.jpg`. Verdict recorded via `mark_asset.py`.

**The headline is that the repaint stage became unnecessary.** Our route existed because
TRELLIS.2's 1024 texture stage bleaches flat-illustration inputs, so we threw its texture
away and repainted with a second model. Pixal3D produces saturated colour in one pass, in
under half the time, and survives retopology to 40k at 3.2 MB.

## Why it is different, and why that predicted the result

From the reverse-engineered spec in `raven38/pixal3d.cpp` (`docs/spec/30-pixal3d-cond.md`):
*"Everything else in the cascade (DINOv3, samplers, SS/shape/tex decoders, postprocess) is
identical to TRELLIS.2."* Same 1.3B DiTs, same tensor counts, same cascade, same VAEs.

One thing changes — how the image conditions each stage:

```
TRELLIS.2:  h = h + cross_attn(h, global_cond)
Pixal3D:    h = h + cross_attn(h, global_cond) + proj_linear(proj_cond)
```

`proj_cond` comes from **projecting the 3D latent grid back into the image**: each latent
voxel is transformed through a real camera (FOV, distance, mesh_scale), projected to a
pixel, and the DINOv3 feature map is bilinearly sampled there. Every 3D token receives the
features belonging to its own location, at low resolution and again through a NAF-upsampled
high-resolution map (hence 2048 conditioning channels on the SLAT stages).

That is a direct mechanism for the bleach: a texture model inferring colour from five global
tokens drifts toward the mean; one sampling the pixel that lands on the surface point has
much less room to. The prediction before running it was "if the drift is a conditioning
problem Pixal3D fixes it, if it is the texture VAE it inherits it" — the VAE is shared, and
the bleach did not survive, so it was conditioning.

It also means Pixal3D **needs a camera** (MoGe-2 estimates the FOV; `--sv-image` synthesizes
a 20° gauge camera) and is **natively multi-view** — four views averaged through per-view
matrices, single-view being the degenerate case. Four turnarounds of a concept is a lever
TRELLIS.2 does not have at all.

## Two Mac ports, and the one to use

**`pawel-mazurkiewicz/Pixal3D-mac` (PyTorch/MPS) does not fit a 32 GB Mac.** Four runs were
killed. The reasons, in order of how much they cost:

- It loads every checkpoint before doing anything: ~22 GB of bf16 at 1024.
- Upstream's `low_vram` mode (on by default) works by `model.to(device)` / `model.cpu()` —
  a **discrete-GPU strategy**. On unified memory both are the same pool, so it frees
  nothing. The feature meant to save small machines is inert on Apple Silicon.
- Its sparse attention falls back to `naive` (materialising the full L×L matrix) because the
  author's `mtlflashattn` shim lives in a different repository and is not vendored.
- The author develops on an M5 Max with **128 GB**.

**`raven38/pixal3d.cpp` (C++/GGML) is the one that works.** Built in four minutes with
`cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` — Metal is automatic on Apple builds,
no flags. Q8_0 weights are 8.09 GB against 24. It compiles real Metal flash-attention
kernels (`kernel_flash_attn_ext_bf16_dk128_dv128`) rather than falling back. The fox ran at
res-1024 in 5:50.

Stage timings from that run, which are worth having as a baseline:

| stage | seconds |
|---|---|
| sparse structure, 12 steps | 77.4 |
| shape SLAT low-res 512 | 17.4 |
| shape SLAT high-res 1024 cascade | 118.4 |
| FlexiDualGrid decode → 5.47M-face mesh | 12.7 |
| texture SLAT flow + PBR decode | 63.3 |
| postprocess → GLB | 24.5 |

### Licence position

Cleaner than the PyTorch route. `pixal3d.cpp` ships **ungated mirrors** deliberately:
`ZhengPeng7/BiRefNet` explicitly as the "RMBG-2.0 substitute", and an ungated DINOv3 mirror,
so no HF token and **no BRIA anywhere**. The PyTorch port needed
`scripts/patch_pixal3d_rembg.py` to achieve the same thing, because the weights'
`pipeline.json` hard-codes `briaai/RMBG-2.0`.

One thing to know: the Q8_0 weight set is published under `license_name: dinov3-license`,
not MIT, because it bundles the DINOv3 encoder. The Pixal3D flow weights themselves are MIT.
Our TRELLIS.2 stack already depends on `facebook/dinov3-vitl16`, so this is not new
exposure — but it is the first time it would be baked into a redistributable bundle.

## What is now open

1. **Only one asset.** The fox is a single subject and the Snag is the hard case; neither
   Pixal3D nor this conclusion has been tested broadly. Same caveat that applies to the
   retopo+repaint route it beat.
2. **Not a controlled comparison.** Different checkpoints, seed and camera convention.
   It shows Pixal3D does not bleach this asset; it does not isolate conditioning as the
   cause beyond the mechanism above.
3. **Q8_0 quantization cost is unmeasured** against the bf16 originals.
4. **Multi-view is untried**, and it is the thing Pixal3D can do that nothing else here can.
5. **The PyTorch port is not dead**, just unusable on 32 GB: the BRIA guard, the model-subset
   patch and the low-VRAM patch all still apply if it is ever run on a larger machine, and
   43 GB of its weights are cached.

## Side finding, and it is not small

The verdict run exposed a bug in `blender_retopo_bake.py`: the Decimate ratio was computed
against `len(mesh.polygons)` while COLLAPSE decimation applies it to **triangles**, and the
voxel remesh emits **quads**. Every face target in this project has been silently doubled —
asking for 40,000 produced 79,991, asking for 20,000 produced 39,361. Fixed, with a test;
a request for 40,000 now lands at 39,987. Face counts in
`journal/retopology-findings.md` should be read as roughly twice what was asked for.
