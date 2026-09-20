# Making TRELLIS.2 faster on Apple Silicon: the attention kernel

How a 1024-cascade run went from 34 minutes to 22, and then to roughly 16, without
changing the model, the sampler, or any quality setting.

Written 2026-09-20. Every number here was measured on this machine; nothing is projected
unless it says so.

## The problem

1024 runs cost too much wall time to iterate on. The bar we set before starting: 10-15
minutes for a good model is an acceptable local trade-off, 20-30 or more is not. Run time
also varies enormously with the input image, so no single measurement is representative.

## Finding the bottleneck without running anything

The first useful thing cost no compute at all. A debug run from 2026-09-03 had kept a full
log with per-step timings, which gave a real 1024 breakdown for free.

| Stage | Share of run |
|---|---|
| Shape SLat, cascade pass | 43% |
| Texture SLat | 24% |
| Bake | 8% |
| Sparse structure | 7% |
| Pipeline load | 6% |
| Decode | 3% |
| Shape SLat, first pass | 3% |

Two sampling passes accounted for 68% of the run. That corrected an assumption we had been
carrying, which treated the texture stage as the thing to optimise; at 1024 the cascade
shape pass is nearly twice as expensive.

## Why sampling is slow: one number

Profiling `SparseMultiHeadAttention.forward` with synchronised timers, against a cached
latent so no earlier stage had to be recomputed:

> **Attention is 93.2% of sampling time.** 65.3s of a 70.0s window, over 120 calls.

Everything else in a sampling step -- sparse convolutions, normalisation, the flow solver
-- is under 7% combined.

The cause is structural. TRELLIS.2-4B is 1536 wide over 12 heads, so its head dimension is
128. PyTorch's MPS backend has no fused attention kernel at any head dimension, and the
vendored Metal kernel supports only up to 64. The model therefore falls back to unfused
SDPA that materialises the full score matrix.

## The fix: MLX has the kernel

MLX's fused attention handles 128-wide heads. Measured at the real Stage-3 shape, 9,801
tokens by 12 heads by 128:

| Backend | fp32 | fp16 |
|---|---|---|
| torch MPS SDPA | 814.9 ms | 868.9 ms |
| MLX fused | 252.0 ms | 59.2 ms |

**Half precision on torch MPS SDPA is slower than fp32.** That refuted a standing
assumption that fp16 was a cheap 2x; it is not a lever on the old path at all. The fp16
win exists only inside the fused kernel.

MLX exposes no zero-copy path from an MPS tensor, so the implementation copies through host
memory on every call. That sounds fatal and is not: including the full round-trip, MLX
still runs 2.83x faster at fp32 and 9.35x at fp16.

## What it delivered on a real asset

Storm Ram, 1024_cascade, seed 0, texture 2048, decimation 300k, against a recorded baseline
with identical parameters. Only the attention backend differs.

| Stage | Baseline | MLX fp32 | MLX fp16 |
|---|---|---|---|
| Pipeline load | 80.9s | 82.6s | 80.3s |
| Sampling, stages 1-3 | 1760.7s | 963.6s | **552.2s** |
| Decode | 53.1s | 53.8s | 59.8s |
| Bake | 161.4s | 241.7s | 165.8s |
| **Whole run** | **2057.4s (34.3 min)** | ~1342s (22.4 min) | **859.5s (14.3 min)** |
| Speedup on sampling | 1x | **1.83x** | **3.19x** |
| Speedup overall | 1x | 1.53x | **2.39x** |

fp16 lands inside the 10-15 minute bar. All three runs used the same seed and parameters.

The fp32 output was inspected and accepted, which is what clears the numerical risk below.

**Attention is no longer the bottleneck.** At fp16 the fixed costs dominate: pipeline load,
decode and bake together are about 306 seconds, 36% of the run, and none of this work can
touch them. Bake is now the single largest stage.

## Using it

The backend is additive and opt-in. The existing `sdpa` path is untouched and remains the
default everywhere, because `mlx` requires setup a fresh clone does not have.

```
uv pip install --python vendor/trellis-space-mac/.venv/bin/python mlx
python scripts/patch_trellis_mlx_attention.py
```

Then `--sparse-attn-backend mlx` on `scripts/trellis_space_generate.py` or
`scripts/trellis_stage3.py`, or the Attention backend control in the web UI's TRELLIS
panel. `I2L_MLX_ATTN_DTYPE=fp16` opts into half precision.

The patch is idempotent and fails closed if upstream anchors move. The attention itself
lives in `image_to_3dlab/mlx_attention.py` so it is importable and unit-tested; the
vendored tree receives only a dispatch branch.

## Caveats, stated plainly

**MLX and torch do not agree exactly, and it does not matter.** At fp32 they diverge by
about 1e-3 on unit-scale inputs, and the gap tracks input magnitude rather than sequence
length. This is *not* the fused kernel: MLX's own matmul-and-softmax diverges by the same
amount, so it is MLX's arithmetic on Metal.

That difference does not survive into a rendered asset. Holding seed, resolution and every
sampler parameter fixed and varying only the attention backend, `sdpa`, MLX fp32 and MLX
fp16 produce visually equivalent output, with face counts inside 0.3% of one another. See
*The attention backend does not change what you get* below.

**One run died between sampling and decode** with
`index -1097849984 is out of bounds: 0, range 0 to 7419814`, raised at the first
synchronisation after decode. Re-running decode from the same cached latents succeeded, so
the fault is transient. The decoder uses convolution blocks and no attention, so the patched
code never runs there. `filter_degenerate_faces` now gathers on CPU rather than boolean-mask
indexing a 7.4M-row tensor on Metal, which was the only MPS operation queued between the two
synchronisation points. **That is a precaution against the likeliest trigger, not a verified
fix.**

**The apparent bake regression was contention, not a regression.** The fp32 run measured
241.7s against the baseline's 161.4s in a stage attention never touches, which looked
alarming. The fp16 run, made with the machine otherwise idle, came in at 165.8s. The fp32
run had diagnostic jobs competing for the GPU. Nothing to fix.

## The attention backend does not change what you get

Worth stating separately, because it is the question that decides whether any of this is
usable, and because it is easy to convince yourself otherwise from confounded comparisons.

**Holding the seed fixed and varying only the attention backend produces visually
equivalent assets.** Three runs at one resolution and one seed — stock `sdpa`, MLX at fp32,
MLX at fp16 — were indistinguishable by eye, with face counts within 0.3%. Neither the
switch to MLX nor the drop to half precision cost anything visible.

![Three renders of the same asset at one seed and resolution, differing only in attention
backend: stock sdpa, MLX fp32, MLX fp16. They are indistinguishable.](images/attention-backends-same-seed.jpg)

*One seed, one resolution, one camera. Rendered with
`scripts/render_glb_comparison.py`, which uses a single shared crop box across the panels
so that identical assets cannot be made to look different by framing.*

Two practical consequences follow:

- **Prefer fp16 wherever MLX is used.** It is not a quality-for-speed trade. fp32 buys
  nothing back.
- **Colour or detail differences between runs are almost never the backend.** They are the
  seed, or the pipeline type, and both are easy to change without noticing.

### The trap this replaced

An early comparison appeared to show the backend washing colour out of an asset. It did
not. The runs being compared differed in *three* ways at once: pipeline type, seed, and
backend. Two of those are enough to change colour substantially on their own, and one of
them is not obvious at all — **each pipeline type uses a different texture model**
(`tex_slat_flow_model_512` versus `tex_slat_flow_model_1024`), so changing resolution
changes which model paints the asset, not merely how finely it is sampled.

The lesson generalises past attention: **when comparing generated output, change one thing.
The web UI randomises the seed unless told otherwise, which silently makes every comparison
a three-variable one.** Pin the seed first, then vary the single thing under test.

## What is left

1. **Route only long sequences through MLX.** About 60 attention calls happen per step and
   many are cross-attention against a short conditioning sequence, where the fixed
   conversion cost dominates. That is why the end-to-end gain was 2.05x rather than the
   kernel's 2.83x. A length threshold recovers most of the gap and changes no numerics.
2. **The gain is resolution-dependent, and at the low end it vanishes.** Attention is
   quadratic in sequence length, so a pipeline type producing ~2,300 tokens does roughly a
   fifteenth of the attention work of one producing ~9,000. At the low end torch's unfused
   path is fast enough that MLX's host round-trip cancels the kernel gain exactly —
   measured as 176s against 184s, inside noise. **MLX is worth selecting only where the
   token count is large.** Any UI that advertises a fixed speedup is wrong at the low end.
3. **Decode faults, twice, both with MLX resident in the process.** Both occurred at the
   higher token count; every low-resolution run decoded cleanly, and re-decoding the same
   cached latents in a fresh process succeeded both times. The decoder uses convolution
   blocks and no attention, so the patched code never runs there — which points at MLX
   holding Metal buffers that decode then lacks, rather than at wrong numbers. Untested
   remedy: release MLX's cache when sampling ends, before decode begins. Not reproduced
   often enough to call diagnosed.
4. **Attack the fixed costs.** At fp16 they are roughly 306 seconds -- 80s pipeline load,
   60s decode, 166s bake -- or 36% of the run, and no attention work can touch them. Bake
   is the largest single stage now.
5. **Reconsider the non-cascade `1024` pipeline type.** TRELLIS.2 supports `512`, `1024`,
   `1024_cascade` and `1536_cascade`; this repo maps `--resolution 1024` to the cascade and
   never exposes plain `1024`, which was abandoned earlier after a 100-minute run. That run
   predates this work, and plain `1024` raises sparse-structure resolution from 32 to 64,
   producing far more tokens -- exactly the regime where the fused kernel wins by the widest
   margin. Measure the token count first, then predict, then decide.
