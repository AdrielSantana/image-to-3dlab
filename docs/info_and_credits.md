# Info & Credits

Built on the shoulder of giants. Thanks to all the hardwork done by folks keeping the Apple Silicon ecossytem alive. 

Markdown counterpart to the Generate page's in-app "Credits & Info" tab
(`viewer/index.html`). The in-app version is the terse, always-current summary; this is
the place for the longer version — more context per pipeline, more room to explain the
tradeoffs. Draft as of 2026-08-18 — expand freely.

## Credits, by pipeline

This repo wraps other people's models and ports. It doesn't train or fine-tune anything
itself (yet — see the fine-tuning notes if that's changed).

### Qwen-Image 2.1 (the Generate Image tab)

**Which model, exactly** — the question worth answering plainly, because "Qwen-Image GGUF"
names half a dozen repositories:

- **Diffusion model:** [`leejet/Qwen-Image-2.1-GGUF`](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF),
  the `Q8_0` file. A straight quantisation of the official
  [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) by the author of
  stable-diffusion.cpp. **This is the stock model, not an uncensored finetune.**
- The [`abenzerps`](https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF) GGUF repository is
  also stock: its metadata says `base_model_relation: quantized`, and its card says "A fully
  uncensored version is currently in development and will be added to this repository soon."
- Running locally, **there is no safety checker anywhere in the pipeline.** The filtering
  people meet on hosted services is a separate layer in front of the model, and
  stable-diffusion.cpp has no such layer. That is why a stock local model can feel
  unfiltered; it is a property of running it yourself, not of a particular repository.
- Changing to a different variant is **one file**: the `--diffusion-model` argument in
  `viewer/image_api.py`. The text encoder and VAE are unchanged.
- **Text encoder:** [`Qwen/Qwen3-VL-8B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
  (`Q4_K_M`). Qwen-Image reads prompts with a vision-language model, which is why the
  encoder is 5 GB on its own.
- **VAE:** [`Comfy-Org/Qwen-Image-2.1`](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).
- **Runtime:** [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)
  (MIT), a prebuilt Metal binary. No ComfyUI.
- **Licence:** Qwen Research License — **non-commercial only**, and it asks that you say
  "Built with Qwen". The restriction sits at the front of the chain, so anything generated
  from one of these pictures inherits it, including after a Hunyuan repaint. Runs land in
  `output/images/research_only/` with a sidecar that says so.

### TRELLIS.2 (clean port)

- **Mac/Metal port foundation:** [pedronaugusto/trellis2-apple](https://github.com/pedronaugusto/trellis2-apple),
  plus Pedro Naugusto's `mtlbvh`, `mtldiffrast`, and `mtlgemm` Metal kernel libraries —
  the pieces that make TRELLIS.2 run on Apple Silicon at all.
- **Upstream model:** Microsoft [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)
  (MIT license; the DINOv3 image encoder it depends on carries its own separate license —
  check before redistributing).
- **The original Mac port** ([shivampkumar/trellis-mac](https://github.com/shivampkumar/trellis-mac))
  is retired internally. It's kept only as the historical source of two self-inflicted bugs
  documented in `CLAUDE.md` (a 200k-face decode cap, and inconsistent mesh winding) —
  not as a build foundation for anything current.
- **Fused attention on Apple Silicon:** [Apple MLX](https://github.com/ml-explore/mlx) (MIT).
  Optional and off by default, selected with `--sparse-attn-backend mlx` or the *Attention
  backend* control in the web UI. PyTorch's MPS backend has no fused attention kernel and
  Pedro's Metal kernel supports head dimensions only through 64, while TRELLIS.2-4B uses
  128 — so the stock path falls back to unfused attention, which was measured at **93.2% of
  sampling time**. Routing it through MLX took a 1024-cascade Storm Ram run from 34.3
  minutes to 14.3. Full method, numbers and caveats:
  [`mlx-attention-2026-09-20.md`](mlx-attention-2026-09-20.md).
- **Input advisor:** [`wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M)
  (MIT). It runs locally after image selection and provides only a conservative warning
  about flat/vector-style inputs. It does not modify the image, block generation, or form
  part of the generated model.

### Stable Fast 3D

- [Stability-AI/stable-fast-3d](https://github.com/Stability-AI/stable-fast-3d)
  (Stability AI Community License). The fast, lower-fidelity option — seconds, not minutes.

### Hunyuan3D-MLX — two variants, two different shape stages

Both variants sit on Tencent's Hunyuan3D-2 model family (Tencent Hunyuan Community
License — **not licensed for use in the EU, UK, or South Korea**; verify exact terms per
model before any redistribution-sensitive use) but combine different people's independent
MLX ports of it. **Since 2026-08-19, they also differ in licensing at the code level, not
just weights** — see the licensing note below.

**dgrauet shape + Xiong paint.**
- Shape stage: [dgrauet](https://github.com/dgrauet)'s MLX port
  (`dgrauet/hunyuan3d-2.1-mlx`, vendored at `vendor/hunyuan-mlx`). Re-verified 2026-08-19
  in a direct A/B against Xiong's own 2.0 shape stage: still the cleanest shape we've
  tested — no dents, no dimples, 10/10 — which is why this path is kept despite the extra
  manual setup below.
- Paint stage: [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX)'s
  paint module, now tracked in-repo at `hunyuan_mlx/paint/` (see below). dgrauet's own
  paint stage produces a shattered, non-coherent UV atlas — that's why paint is sourced
  from a different repo entirely rather than staying single-author.
- **Licensing:** dgrauet's shape code carries Tencent's Community License, not a
  permissive one (all three of its `LICENSE` files are Tencent's own text) — the same
  territorial/use restriction that applies to the weights applies to the *code*, too.
  It stays manually vendor-cloned (`vendor/hunyuan-mlx`) rather than brought into this
  repo's tracked tree.
- Wired up via `scripts/hunyuan_mlx_generate.py`.

**Xiong, full pipeline** — both shape and paint from the same repo, one author, end to end.
- [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX) — `hy3d shape`
  and `hy3d paint` under a shared codebase, parity-tested against the original PyTorch
  reference and against a native Swift port in the same repo. **MIT licensed.**
- **Brought in-repo 2026-08-19**: the code (not weights) moved from
  `vendor/hunyuan-mlx-paint` into this repo's tracked tree at `hunyuan_mlx/shape/` and
  `hunyuan_mlx/paint/` (MIT notice preserved at `hunyuan_mlx/LICENSE`). A clone of this
  repo alone has the code that runs; only `weights/` (multi-GB, git-ignored) needs a
  separate download — `python hunyuan_mlx/download_weights.py` pulls them from Hugging
  Face. `uv sync` in each of `hunyuan_mlx/shape` and `hunyuan_mlx/paint` sets up the venvs.
- **Model choice, benchmarked 2026-08-19** (Flicker, octree=512, quantize=8, 30 steps,
  shape stage only): **2.0** ~167s, cleanest result, Xiong's own recommended pick and now
  this app's default; **2.0-turbo** ~60-105s but shows real distillation-noise dents even
  at 30 steps (its PCM schedule caps out at 100 steps — more steps helps, doesn't fully
  clear it); **2.1** ~450s with `--octree-decode` (~48min without) and not Xiong's
  recommended pick regardless (weaker DINOv2-large conditioner vs 2.0/2.0-turbo's
  DINOv2-giant). Full writeup: `docs/hunyuan-mlx-recipes.md`.
- Independently verified clean (Blender Face Orientation overlay, no flipped winding).
- Wired up via `scripts/hunyuan_mlx_xiong_generate.py`.

### Evaluated, not shipped

- [RobertBeckebans/AI_trellis2cpp](https://github.com/RobertBeckebans/AI_trellis2cpp)
  (C++/ggml Metal port). A real upstream `purego` ARM64 bug was found and reported while
  testing it (mis-packed stack-spilled arguments on Apple's tight per-type ABI packing —
  matches `ebitengine/purego#352`/`#353`, fixed upstream in v0.10.0+).

## Speed

A 1024 run is dominated by attention. The *Attention backend* control decides how that work
is done, on the same input, seed and parameters:

| Attention backend | at 1024 cascade | at 512 | Setup needed |
|---|---|---|---|
| `sdpa` (default) | 34.3 min | 5.4 min | none |
| `mlx` (fp32) | 22.4 min | no change | mlx in the backend venv + the patch |
| `mlx-fp16` | **14.3 min** | no change | same |

**MLX is only worth selecting at 1024 and above.** Attention cost grows with the square of
the token count, and 512 produces roughly a quarter of the tokens, so the stock path is
already fast enough there that the fused kernel's advantage is cancelled by the cost of
moving tensors into MLX and back. Measured at 512: 176 seconds against 184, inside noise.

**The choice does not change the output.** At a fixed seed and resolution, `sdpa`, `mlx`
and `mlx-fp16` produce visually identical assets, with face counts within 0.3%. So prefer
fp16 wherever MLX is used; fp32 buys nothing back. See
[`mlx-attention-2026-09-20.md`](mlx-attention-2026-09-20.md).

Both MLX options need one-time setup, and the Generate page's Setup card reports whether
they are available and names whatever is missing:

```
uv pip install --python vendor/trellis-space-mac/.venv/bin/python mlx
python scripts/patch_trellis_mlx_attention.py
```

One honest qualification: run time varies enormously with the input image, so the figures
above are one worked example rather than a promise.

## Known shortcomings

As of 2026-08-19. This list is honest-and-incomplete on purpose — update it as things
change rather than letting it go stale.

- **Texture tear on concave geometry (inner thigh, armpit, ear folds) — fixed
  2026-08-19.** The paint stage filled texels no camera could see (self-occluded creases)
  by grabbing the nearest already-painted texel in flat 2D UV-atlas space —
  xatlas can and does pack unrelated 3D regions (an eye chart, a leg chart) next to each
  other on that flat sheet, so occluded creases got filled with the wrong, unrelated
  color. Root-caused by measuring true camera occlusion directly: 7.8% of surface texels
  had zero visibility from all 6 fixed views, clustered into ~7 localized regions (a real
  occlusion signature, not rasterizer noise). Fixed by filling occluded-but-in-chart
  texels from their nearest neighbor in actual **3D surface space** instead of 2D atlas
  space. Applies to *both* Hunyuan variants — they share the same paint stage. The
  shape-stage geometry fusion defect noted previously is a separate, still-open issue.
- **Hunyuan's paint stage has a hard face-count wall.** The `xatlas` UV-unwrap step goes
  from ~3 minutes to 37+ minutes between 500k and 700k faces; 1M faces never completed in
  testing. Keep `decimation_target` at or under 500,000. This applies to *both* Hunyuan
  variants — they share the same paint stage.
- **TRELLIS.2 material generation can drift from the reference image.** Community reports
  and our controlled tests show a particularly severe failure mode for some flat/vector
  illustrations: their predicted base colour can become nearly black. The same inputs fail
  on MPS and official CUDA, while 3D-rendered versions preserve their colour, so this is not
  a Metal-port artefact. TRELLIS.2 *is* directly conditioned on DINOv3 image features as
  well as generated geometry; it does not copy source pixels onto the mesh. Full evidence,
  user guidance, and next steps are in
  [`trellis2-flat-illustration-colour-drift.md`](trellis2-flat-illustration-colour-drift.md).
  The Generate page now pairs a static warning with a local TinyCLIP advisory. Its score
  is similarity-based rather than a calibrated failure probability, so manual inspection
  remains necessary.
- **dgrauet's shape stage stays manually vendor-cloned** (`vendor/hunyuan-mlx`) — it's
  Tencent-licensed *code*, not just weights, so it isn't part of the clone-and-go
  simplification below. Xiong's shape+paint is MIT and tracked in-repo at `hunyuan_mlx/`:
  `uv sync` in `hunyuan_mlx/shape` and `hunyuan_mlx/paint`, then
  `python hunyuan_mlx/download_weights.py` (Hugging Face). RealESRGAN super-res weights
  aren't part of the official HF repos, so they're a separate step:
  `hunyuan_mlx/paint/scripts/convert_realesrgan.py` (needs torch, dev-time only) fetches
  the official [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) release and
  converts it — bit-identical to what this repo had shipped without a documented source
  before 2026-08-19.
- **Xiong, full pipeline — benchmarked 2026-08-19** (Flicker, octree=512, quantize=8,
  30 steps, shape stage only): 2.0 ~167s (default, cleanest); 2.0-turbo ~60-105s (real
  distillation-noise dents even at 30 steps); 2.1 ~450s with `--octree-decode` (~48min
  without), not Xiong's recommended pick regardless (weaker DINOv2-large conditioner).
  Full writeup: `docs/hunyuan-mlx-recipes.md`.
- **The retired TRELLIS Mac port** shipped two self-inflicted bugs for a long time before
  they were caught: a 200,000-face cap that crushed every decode with a crude decimator,
  and inconsistent mesh winding that left assets hollow under backface culling. Full story
  in `CLAUDE.md`. Worth remembering as a cautionary tale even though that port is retired —
  a hollow, backwards-facing mesh looks completely fine in a double-sided glTF preview and
  only fails once something backface-culls it.

## Power-user notes

Peeking behind the curtain without the browser:

- A submitted job's folder appears in `output/` the instant you hit Generate, before any
  compute starts — `ls -lat output/ | head` confirms a job was accepted.
- `tail -f output/<folder>/run.log` streams the exact same progress lines the browser's
  live view shows.
- `ps aux | grep -E "trellis_space_generate|hunyuan_mlx_generate|hunyuan_mlx_xiong_generate|pipeline.py"`
  confirms the generation subprocess is alive and shows its exact arguments.
- Every TRELLIS run checkpoints `<out>_latents.pt` immediately after sampling and before
  decode. If decode or bake fails, that checkpoint is retained and can be resumed with
  `--from-latents`, even when `debug` is off. After a successful non-debug run it is
  removed along with the other diagnostic artifacts; enable `debug` to retain the
  manifest, latents, decoded mesh, textures, and intermediate meshes after success.

