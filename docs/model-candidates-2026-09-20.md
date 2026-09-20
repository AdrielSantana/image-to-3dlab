# Image → 3D model candidates: CUDA blockers and Mac ports

Researched 2026-09-20. Six candidate backends were assessed for whether they can run on
Apple Silicon at all, and at what porting cost. Nothing here has been run yet — this is a
desk check of dependencies, licences and existing ports, done before spending a run.

**Pixal3D was added later the same day** and changes the shape of this list: it is the only
candidate with a finished Apple Silicon port, and that port shares our kernels. Read its
section before the others — two of its bug fixes apply to code we are running right now,
whether or not we ever generate a single mesh with it.

The question that decides everything: **does the repo depend on a CUDA-only compiled
extension, or is it plain PyTorch?** Plain PyTorch usually moves to MPS with a device
change. A compiled CUDA kernel with no Metal equivalent means writing one.

## Verdict table

| Repo | Licence | CUDA blocker | Mac port | Cost to try |
|---|---|---|---|---|
| [TripoSG](https://github.com/VAST-AI-Research/TripoSG) | MIT | none in the inference path | none found | **low** |
| [Hi3DGen](https://github.com/bytedance/Hi3DGen) | MIT | `spconv` (hard) | none found | high |
| [InstantMesh](https://github.com/TencentARC/InstantMesh) | Apache-2.0 | `nvdiffrast` | none found | medium |
| [Unique3D](https://github.com/AiuniAI/Unique3D) | MIT | `nvdiffrast`, `pytorch3d`, `torch_scatter`, `onnxruntime_gpu` | none found | high |
| [Hi3D-Official](https://github.com/yanghb22-fdu/Hi3D-Official) | MIT | none named, but 80 GB A100 reference | none found | high |
| [Pixal3D](https://github.com/pawel-mazurkiewicz/Pixal3D-mac) | MIT | the whole CUDA stack — **already ported** | [Pixal3D-mac](https://github.com/pawel-mazurkiewicz/Pixal3D-mac) | **lowest of the six** |

## Per-repo detail

### Pixal3D — the only one already running on a Mac, and it shares our kernels

Pixel-Aligned 3D Generation from Images: SIGGRAPH 2026, from Tsinghua, Tencent ARC Lab and
Victoria University of Wellington. Single image in, textured GLB out. Descended from the
TRELLIS family by way of Direct3D-S2, weights on Hugging Face at `TencentARC/Pixal3D`,
MIT throughout. The Mac port is `pawel-mazurkiewicz/Pixal3D-mac`, and the porting story is
written up at
[blog.chillaid.art](https://blog.chillaid.art/posts/porting-pixal3d-one-cursed-kernel-at-a-time).

Requirements match ours almost exactly: Apple Silicon, macOS 14+, Xcode command line tools,
Python 3.10, six vendored Metal packages editable-installed from `extern/`. A 1024-cascade
generation takes "several minutes"; 1536-cascade at 2048px texture took up to 50 minutes on
an M5 Max. The interface is one command:

```bash
.venv/bin/python generate_mps.py image.png --output out.glb \
    --pipeline-type 1024_cascade --texture-size 2048 --seed 0
```

**The reason to read this before trying anything else** is not the model. It is that the
port is built on Pedro Naugusto's Metal reimplementations — `cumesh` → `mtlmesh`,
`cubvh` → `mtlbvh`, `nvdiffrast` → `mtldiffrast` — which is **the same lineage as the
kernels under `vendor/trellis-space-mac/deps/`**. Its bug fixes are therefore not
inspiration, they are patches to code we execute on every run.

Two in particular:

**1. The simplifier's atomic race, and 92.6% non-manifold edges.** One Metal kernel writes
a buffer as `device atomic_ulong*`; the next reads the same buffer as `device const
ulong*`. That mismatch is undefined behaviour under the Metal Shading Language spec, and
in practice the reader gets stale values: two threads decide to collapse the same edge and
corrupt the topology. Measured by that author, **92.6% non-manifold edges against CUDA's
5.18%**. Declaring the reader `device const volatile ulong*` forced fresh loads and took it
to **5.15%** — CUDA parity from a one-word change.

Our copy has the unfixed declaration, at
`vendor/trellis-space-mac/deps/mtlmesh/src/metal/simplify.metal:137` (writer) and `:163`
(reader), with no `volatile` anywhere in the file. It is on the live path:
`trellis2/representations/mesh/base.py` calls `import cumesh`, and `cumesh` resolves to
`deps/mtlmesh/cumesh/`, which the 2026-09-20 Snag run log confirms.

**2. PyTorch's MPS fused attention returns garbage above ~18,000 tokens.** That author
measured `scaled_dot_product_attention` on MPS as bit-accurate to roughly 18-20k tokens and
wrong above it — mean absolute error 0.05-0.10, no warning, no error. The stated symptom is
"shrunken, desaturated geometry". Their workaround is a manual chunked backend: 2048-query
chunks, matmul → softmax → matmul, in fp32.

We cross that threshold routinely — the Snag samples at 23,025 tokens — and our own runs
still report `[ATTENTION] Using backend: sdpa` for the dense path even now that sparse
attention runs on MLX. Note that `journal/decode-crash-debug.md` records token count as a
*dead* hypothesis: that was for decode **crashes**, and this is silent corruption, which
was never tested. It is also worth holding next to
`docs/trellis2-flat-illustration-colour-drift.md`, where the 1024 fox came out bleached.

### TripoSG — the one to try first

Rectified-flow transformer over a signed-distance-function VAE. Geometry only, no texture,
which is exactly the shape half of a two-step.

`requirements.txt` lists `diso`, which is CUDA-only by construction and has no CPU or
Metal build. **But `scripts/inference_triposg.py` never imports it.** The mesh comes back
from the pipeline as plain vertex and face arrays and goes straight into `trimesh.Trimesh`.
So the CUDA-only dependency is declared but unused on the inference path, and can simply
not be installed.

What does need changing is small and known:

- `device = "cuda"` is hardcoded in the inference script.
- Background removal calls `BriaRMBG` against `briaai/RMBG-1.4`. This repo uses `rembg`
  already, so that call gets swapped rather than ported.
- `numpy==1.22.3` is pinned unusually low and will fight a modern Mac stack.

Everything else is `diffusers`, `transformers`, `einops`, `trimesh`, `pymeshlab`.

### Hi3DGen — blocked, not merely expensive

Generates geometry via an intermediate normal map, and the README install line is
`pip install spconv-cu{your-cuda-version}==2.3.6 xformers==0.0.27.post2`.

`spconv` is the blocker and it is a hard one. Its only non-CUDA build is a CPU wheel that
upstream describes as Linux-only, unoptimised, and for debugging. There is no Metal
backend and no Apple Silicon support of any kind. Sparse convolution is the load-bearing
part of the architecture, not a detail at the edge, so there is no small patch that routes
around it.

Note the mismatch with existing work here: Hi3DGen sits on the TRELLIS sparse backbone,
while this repo's Metal effort targets TRELLIS.2, a later and differently shaped model.
The kernels do not transfer for free.

### InstantMesh — medium, and probably not worth it

`requirements.txt` pulls `git+https://github.com/NVlabs/nvdiffrast/` directly. nvdiffrast
is a CUDA rasterizer with no Metal build. It is used for rendering and texture baking, so
the vertex-colour mesh path may avoid it while `--export_texmap` cannot.

Since this repo would take geometry only and hand it to the existing paint stage, the
nvdiffrast path might be dodged entirely. The reason to deprioritise it is quality, not
feasibility: it is an older transformer-plus-multiview design and its geometry is weaker
than the flow-based models above.

### Unique3D — worst Mac fit of the five

Stacks `nvdiffrast`, `pytorch3d` from source, `torch_scatter`, `onnxruntime_gpu`,
`ort_nightly_gpu`, `rembg[gpu]` and `xformers`. Four separate CUDA-compiled dependencies,
any one of which is a porting project. Skip.

### Hi3D-Official — skip

Video-diffusion approach built on Stable Video Diffusion plus 3D Gaussian Splatting, with
a DPT depth model and CLIP. Authors tested on an 80 GB A100. No custom CUDA extension is
named, but the memory profile and the multi-model chain make it the heaviest and oldest
option here. Nothing suggests it beats the alternatives on output quality.

## Not a lane: replacing the TRELLIS.2 or Hunyuan ports

Decided 2026-09-20. Both ports are settled and we are not redoing that work. An
MLX-native TRELLIS.2 rewrite exists ([lyonsno/trellis2mlx](https://github.com/lyonsno/trellis2mlx),
MIT, claiming ~3-5 GB against ~40-55 GB) and is recorded here only so it is not
re-researched. It is not on the roadmap.

The open TRELLIS.2 complaint is **render time at 1024 on this Mac**, which is a
performance problem inside the port we already have, not a reason to adopt another one.
The known lever for it is the Stage-3 attention cost, which is attention-bound rather than
compute-bound. That is a separate lane from this bake-off.

## Suggested order

1. **TripoSG** into the existing paint stage, as the first real two-step. Geometry from a
   model with no CUDA blocker, texture from the paint stage we already run.
2. Reassess. Hi3DGen is the most interesting geometry model of the five on paper and the
   only one that is genuinely blocked. Revisit only if a Metal sparse-convolution path
   appears upstream.

## Method note

Licences and dependencies were read from each repo's own `requirements.txt` and README
rather than from summaries. Searches for Apple Silicon, MPS and MLX ports of TripoSG,
Hi3DGen, InstantMesh and Unique3D returned nothing for any of the four.
