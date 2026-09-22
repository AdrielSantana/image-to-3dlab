# Qwen detour: a local image model at the front, a local VLM at the Blender end

**Date:** 2026-09-22. **Status:** evaluation only. Nothing downloaded, nothing installed,
no code changed. Two questions, answered separately because they turn out to share a
component.

---

## 1. Qwen-Image-2.1 as the front of the pipeline

### What the linked repo actually is

The link was `abenzerps/Qwen-Image-2.1-Uncensored-GGUF`. The HF API resolves that id to
**`abenzerps/Qwen-Image-2.1-GGUF`**: created 2026-09-20, 33k downloads, 677 likes. The card
says an uncensored variant is *in development*; the files present are straight quantisations
of the stock base model. So the specific thing linked is not (yet) an uncensored model.

Its file list, which matters more than the headline:

| File | Size |
|---|---|
| `qwen-image-2.1-Q4_K_M.gguf` (recommended) | 4.60 GB |
| `qwen-image-2.1-Q8_0.gguf` | 7.59 GB |
| `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 9.35 GB |
| `text_encoders/qwen3vl_8b_bf16.safetensors` | 17.53 GB |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB |

**A working Q4 set is ~14.6 GB, not 4.6 GB.** The "4.6 GB model" number is the DiT only.

Note the text encoder: **Qwen-Image-2.1's text encoder is Qwen3-VL-8B**. Question 1 and
question 2 want the same weights on disk.

### The base model

`Qwen/Qwen-Image-2.1`, released 2026-09-14. 7.1B single-stream block-causal DiT with a
Qwen3-VL text encoder, 40-step guidance-free sampling. Three properties are relevant here:

- **Native RGBA.** Four-channel output with a real alpha channel, straight from the model.
- **Native 2K.** 2048x2048 direct, not upscaled.
- **Generation and editing in one checkpoint**, with multiple reference images (the Comfy
  encode node opens up to 16 image inputs).

### The blocker-shaped fact: the licence

`Qwen/Qwen-Image-2.1` is `license: other`, `license_name: qwen-research`. The **Qwen Research
License is non-commercial only**, plus a "Built with Qwen" attribution requirement.

This repo treats licence provenance as first-class, and `image_to_3dlab/provenance.py`
currently knows two non-clear classes: `commercial-conditional` (SF3D, TRELLIS) and
`territory-restricted` (Hunyuan). A non-commercial model is a **third class we do not have**,
and it is stricter than either. Worse, it sits at the *front* of the chain: a 3D asset
generated from a Qwen-Image source image is downstream of a non-commercial model's output,
so the taint propagates to everything after it, including anything Hunyuan repaints.

Two ways out, if commercial use ever matters:

- **Qwen-Image 1.0** (Aug 2025, 20B) is Apache-2.0. Costs you the alpha channel, the 2K
  native output and the 2.1 quality, and it is 3x the parameters.
- Keep 2.1 strictly in a `research-only` lane with its own output folder and a provenance
  classification that `validate_run_policy` refuses by default, the same way
  `allow_conditional` works today.

### The route on this Mac

GGUF here means **ComfyUI + ComfyUI-GGUF** (the card points at the leejet fork). We
deliberately left the ComfyUI route behind. The native route is MLX:

- **mflux** has ported Qwen-Image-2.1 as a pure-MLX backend (`mflux-generate-qwen21`).
- MLX weight conversions exist already: `SirSahOl/Qwen-Image-2.1-mlx-{4bit,8bit,16bit}`,
  `toxicdog/Qwen-Image-2.1-MLX`.
- Third-party measurement, **not verified here**: M5 Max, bf16, 1600x672, ~1.78 s/step plus
  ~8 s fixed overhead. 10 steps ~26 s, 40 steps ~79 s. This machine is a 32 GB Mac17,2, so
  expect 8-bit or 4-bit and slower than that, but minutes per image, not hours.

**Recommendation: if we do this, do it through mflux/MLX, not GGUF/ComfyUI.** Same model,
no ComfyUI regression, and it lines up with the MLX work already done for TRELLIS.2.

### Will it run on a 32 GB Mac? Yes, and memory is not the problem.

Measured figures from the `mlx-serve` Qwen-Image-2.1 quantized packs (PR #477), which are
sized explicitly for 32 GB and 16 GB Macs:

| Preset | On disk | **Peak process footprint** | Measured run |
|---|---|---|---|
| 8-bit (the 32 GB preset) | 17.6 GB | **12.95 GB** | 1024x1024, 40 steps, ~23 s/step, ~985 s wall clock incl. load |
| 4-bit (the 16 GB preset) | 10.7 GB | **9.55 GB** | 1024x1024, 3 steps ~87 s; 512x512, 20 steps ~118 s |

Peak 12.95 GB on a 32 GB machine leaves roughly 19 GB free. It will not kill anything.

What makes that possible is a **staged text encoder**: the 8B encoder is loaded per request,
runs the prompt, and is freed before denoising starts, so the DiT and the encoder are never
resident together. Both packs also keep the VAE, embeddings, norms and small shared linears
at f32 and quantise only the DiT blocks and encoder layers. Any route we take needs to do
the same thing; a naive load of DiT + encoder + VAE at 8-bit is ~17 GB resident before a
single activation.

**Speed is the real cost, and the public numbers disagree.** `mlx-serve` reports ~23 s/step
at 8-bit/1024. Ivan Fioravanti reports ~1.78 s/step at bf16 on an M5 Max at 1600x672, i.e.
40 steps in ~79 s. Different chip, different quantisation, different resolution, and
quantised MLX inference carries dequantisation overhead that bf16 does not. This machine is
a 32 GB Mac17,2 and will land somewhere between. **Unverified: measure it before believing
either number.**

One hard constraint: **do not run the image model and a 20 GB LLM at the same time.**
13 GB + 20 GB does not fit. These are two separate sessions.

### Why it is a genuinely good fit

1. **RGBA out of the model deletes the background-removal step.** `trellis_backend.py:218`
   runs `rembg`/u2net whenever the input alpha is fully opaque, and SF3D calls
   `remove_background` unconditionally. A source image born with alpha removes a provenance
   component, a dependency, and a whole class of silhouette damage. u2net eating thin
   structures is exactly the shape of the old foliage/hole trouble.
2. **Multi-reference editing is the multi-view lever we already owe.** `docs/resume-guidance.md`
   names multi-view conditioning as the real fix for the warrior girl's blade direction.
   Feeding one source image plus a "same character, three-quarter view from the left" edit
   prompt is a plausible way to synthesise the extra views. **Caveat: this is not a
   view-consistent multi-view model** (not Zero123 / MV-Adapter). Cross-view identity
   consistency is a hypothesis to test, not a property to assume.
3. **Dataset generation for the parked TRELLIS fine-tune** (see `trellis-can-be-fine-tuned`),
   where non-commercial licensing of the image generator is much less awkward.

### The honest counterargument

It does not fix anything that is currently broken. The open list is the decode crash at
1024, geometry fidelity, retopo/repaint validation across assets, the eyes. "We cannot get
source images" is not on it. And there is a subtler cost: **generating the source undercuts
"closeness to source" as the north star.** When the input is synthesised too, there is no
ground truth to be close to, and quality judgement goes circular.

### Verdict

Worth doing, but as its own opt-in front-end stage with its own licence class, not as a
default. The value is the alpha channel and the multi-view experiment, not "we can make
pictures now". Smallest useful slice:

1. Add a `research-only` / non-commercial classification to `provenance.py` and make
   `validate_run_policy` refuse it unless the manifest opts in. Test-first, cheap, and it is
   owed regardless of whether we ever install the model.
2. Install mflux + the 8-bit MLX conversion, announced with its real size.
3. One experiment, one question: generate a creature with alpha, run it through TRELLIS.2
   and pixal3d, and compare against the same creature via the current rembg path.
4. Only then try the multi-view edit and see whether the views are consistent enough to
   condition on.

---

## 2. Qwen3-VL-8B driving Blender

### The relevant evidence

There is a direct measurement, published 2026: **VIGA / BlenderBench**
(arXiv 2601.11109, "Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning").
BlenderBench has three tasks: camera adjustment, fixed-view editing, exploratory editing.
It reports Photometric Loss (lower better), Negative-CLIP (lower better) and a VLM Score
(higher better). Qwen3-VL-8B is one of the four evaluated models, alongside GPT-4o,
Claude-Sonnet-4 and Gemini-2.5-Pro.

VLM Score, Tasks 1 / 2 / 3, best-of-1:

| Model + setting | T1 | T2 | T3 |
|---|---|---|---|
| Qwen3-VL-8B, one-shot raw bpy | 0.28 | 1.61 | 1.25 |
| Qwen3-VL-8B, inside the VIGA loop | **1.31** | **3.33** | **2.25** |
| GPT-4o, one-shot raw bpy | 0.58 | 2.75 | 0.25 |
| GPT-4o, inside the VIGA loop | 1.44 | 3.58 | 1.53 |
| GPT-4o, VIGA best-of-4 | 3.25 | 3.83 | 1.61 |

Read that carefully, because it is the whole answer:

- **One-shot, an 8B model writing raw bpy is bad.** Hallucinated API calls, wrong parameter
  syntax, non-executable loops. That is the paper's own list of small-model failure modes.
- **Inside a proper loop, the same 8B model is roughly GPT-4o-class on these tasks**, and on
  Task 3 (explore the scene *and* make multi-step edits) it beats every GPT-4o setting
  measured, including best-of-4. The improvement is +264% / +26% / +312%.
- **The loop, not the model, is the variable.** Two ingredients do the work: a render →
  compare → revise cycle with the rendered result fed back as an image, and explicit state
  memory so later edits do not overwrite earlier ones.
- **And the paper does not hand the model raw bpy.** Its Table 1 is a *high-level skill
  library*, split into State Observation (look, without changing anything) and State
  Modification (change something, persistently). That split is the fix for hallucinated API
  calls.

### Where the 8B model still loses

On BlenderGym, which is fine-grained single-step editing, Qwen3-VL-8B gains only **+22.6%**
from the loop, against Gemini-2.5-Pro's +47.4%. On the Blend Shape category it gets
**worse**: photometric loss 7.76 one-shot → 13.51 with the loop. Small model, subtle
parametric edit, and the feedback loop actively misleads it.

That maps uncomfortably onto our work. Coarse and structural ("the camera is too low", "the
key light is on the wrong side", "the limb is clipping the body") is the regime where it
works. Our actual daily edits are weights, blend shapes, iris placement, bind poses, which is
the regime where it degraded.

### Why this repo is unusually well placed anyway

We already built the thing the paper says is the deciding factor. There are ~40
`scripts/blender_*.py` tools, the live Blender socket on port 9876, and a render-and-judge
side that already exists: `blender_render_asset.py`, `blender_inspect.py`,
`blender_pick_pixel.py`, `compare_to_source.py`, `blender_turntable.py`. The Observation /
Modification split is there de facto. What is missing is only a tool-call schema over those
scripts and a driver that runs the loop.

That is a meaningful head start: the expensive half of VIGA is the skill library, and ours is
better than a benchmark's because it was written against real assets.

### What else could drive Blender locally, including what is already on disk

Qwen3-VL-8B is the model the benchmark measured, which is the only reason it is the
reference point. It is a year old (Sept 2025), and it is not the best local option available
here. Six candidates, five of them already downloaded:

| Model | On disk | Sees a render? | Notes |
|---|---|---|---|
| **`gemma-4-31b-it-qat`** | **19.6 GB, already here** | **Yes** (`Gemma4ForConditionalGeneration`, vision config present) | Native function calling. Tool-calling jumped from 6.6% in Gemma 3 to **86.4%** in Gemma 4; LiveCodeBench v6 **80.0%**, Codeforces ELO 2150. 256K context. Sees *and* codes. |
| `gemma-4-12b-it-qat` | 7.15 GB, already here | Yes (`mmproj` file present) | Same family, same abilities, a third of the footprint. Leaves plenty of room for Blender. |
| `ornith-1.0-35b` | 21.2 GB, already here | No | MoE, ~3B active. Purpose-built for **agentic coding**, MIT licensed, SOTA at its size on Terminal-Bench 2.1 / SWE-Bench. Notable quirk: it learns its own scaffold during RL rather than relying on a fixed harness, which is interesting given our harness is the thing we would be writing. Blind. |
| `qwen3-coder-30b-a3b-instruct-mlx` | 21.0 GB, already here | No | Strong code and tool calls. Blind. |
| `gpt-oss-20b` | 12.1 GB, already here | No | Decent tool calling. Blind. |
| `Qwen3-VL-8B-Instruct-4bit` | ~5 GB, **not here** | Yes | The benchmarked model. Weakest coder of the lot. |

**The conclusion is inconvenient for the original question: we should probably not use
Qwen3-VL-8B.** `gemma-4-31b-it-qat` is already on disk and is better on both axes that
matter, vision *and* code generation. The 8B's measured weakness in VIGA was exactly code
generation, and Gemma 4 31B is several tiers above it there while still being multimodal.

The honest caveat: **nobody has published a BlenderBench number for Gemma 4.** The VIGA
result transfers as an argument, not as a measurement. But the paper's own finding is that
the loop and the skill library dominate the model choice, and on the model axis Gemma 4 31B
strictly dominates Qwen3-VL-8B.

**Memory, running alongside Blender.** Gemma 4 31B QAT is 19.6 GB of weights plus KV cache;
Blender with a mid-size GLB and a render buffer is 2-4 GB. That is ~25 GB of 32 with the
context capped well below 256K. Workable, not comfortable. Gemma 4 12B at 7.15 GB is
comfortable and keeps the same capabilities.

**Plan: start on `gemma-4-12b-it-qat`, escalate to 31B only when the 12B is demonstrably the
limiter rather than our tool schema.** Iterating on the harness is much faster at 7 GB, and
the first failures will be ours, not the model's.

### Practicalities on this machine

- 32 GB, Mac17,2, 433 GB free, Blender 4.x installed, LM Studio installed with six models.
- **Cost per turn.** Every iteration is a Blender render plus a vision pass; call it 20-60 s
  on-device. The VIGA config allows `max_iterate_round=100`. This is an overnight-agent
  shape, not an interactive one. Budget accordingly, and cap the round count low while the
  harness is still being debugged.
- **Fallback if one model cannot do both jobs: split the roles.** A vision model looks at the
  render and says what is wrong in words; `qwen3-coder-30b-a3b-instruct-mlx` or
  `ornith-1.0-35b` writes the script against our tool schema. Fully local. But do not start
  here: two models means two failure surfaces, and Gemma 4 is supposed to do both.

### Verdict

Worth a spike, and a cheap one, because the skill library already exists. Scope it to the
coarse tasks where a small local model measurably works: staging, camera, lighting, coarse
placement. Do not expect it to do rigging, weights or blend shapes.

Suggested first test, **which needs zero downloads**:

1. Start the LM Studio server on the already-installed `gemma-4-12b-it-qat` (7.15 GB,
   multimodal, native function calling).
2. Define a 6-8 tool schema over existing scripts: import asset, set camera, set light,
   render, read render, move object, report scene state.
3. Task: "stage and light this GLB for a turntable so the face reads clearly." Judge by the
   render.
4. If the 12B can drive our tools at all, close the loop by feeding the render back as an
   image, then escalate to `gemma-4-31b-it-qat` only if the 12B is demonstrably the limiter.

If step 1 fails, no vision model would have saved it, and we spent an hour.

---

## Downloads this would require (nothing fetched yet)

Per the no-surprise-downloads rule, stated up front:

| Thing | Route | Size |
|---|---|---|
| Qwen-Image-2.1, 8-bit MLX | mflux | ~9-10 GB |
| Qwen-Image-2.1, 4-bit MLX | mflux | ~5-6 GB |
| Qwen-Image-2.1 Q4 GGUF set (DiT + int8 text encoder + VAE) | ComfyUI-GGUF | ~14.6 GB |
| Qwen3-VL-8B-Instruct, 4-bit MLX | mlx-vlm / LM Studio | ~5 GB |

The Blender spike needs **none of these**. `gemma-4-12b-it-qat` and `gemma-4-31b-it-qat` are
already on disk and cover it.

## Sources

- https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF (HF API metadata, 2026-09-22)
- https://huggingface.co/Qwen/Qwen-Image-2.1 (licence: `qwen-research`)
- https://comfy.org/qwen-image-2.1/
- https://github.com/QwenLM/Qwen-Image-2.1
- https://github.com/filipstrand/mflux
- arXiv 2601.11109, VIGA / BlenderBench
- https://x.com/ivanfioravanti/status/2101257247292580045 (M5 Max MLX timings)

---

## Addendum: it was built and run the same day

`scripts/blender_agent.py` (14 verbs, 79 tests) now exists, and three live runs went through
the real Blender socket on 9876 with `gemma-4-31b-it-qat`. The evaluation above predicted
the outcome closely enough to be worth recording.

**What held up.** The argument that the loop and the skill library matter more than the
model was correct, and so was the prediction about *where* a local model succeeds. Asked for
a red cube on a lit ground plane, the model got it right immediately. Asked for a Rubik's
cube, it produced a clean 3x3x3 grid with correct gaps, framed it three-quarter and lit it,
and its own running commentary between rounds was "The scene is empty", then "too dark to
see anything clearly", then "too large for the frame and lacks color", each followed by a
correct targeted fix. That is the interleaved loop working exactly as described.

**What failed was the predicted thing.** The face colouring is wrong. It repeatedly
recolours edge cubelets because it confuses which grid index is "top" and which is "right".
Coarse and structural, good; precise index reasoning, poor. This is the BlenderGym finding
reproduced on our own bench, one working day after reading it.

**Model choice was decided by measurement, not by the table above.** `gemma-4-12b-qat` is
**unusable for this**: LM Studio cannot render tool schemas into its Jinja template and
every request carrying `tools` returns HTTP 400. Vision and plain chat work on it; tools do
not. The 31B works for tools, and for tools plus an image in the same request.

**Four bugs that only a live run could find**, all now regression-tested:

1. **Parallel tool calls break the interleave.** The model batches modifications and skips
   rendering no matter what the system prompt says. The first run made eight calls in one
   round and rendered zero times. The harness now forces a render after any round that
   changed the scene without looking at it. *Prompting a behaviour is not enforcing it.*
2. **A relative output directory silently broke every render.** The generated code runs
   inside Blender, whose working directory is not ours, so renders failed with a read-only
   filesystem error while every other tool succeeded.
3. **Linked duplicates share a mesh datablock, and material slots live on the mesh.** The
   first Rubik's cube came out uniformly blue: each colour repainted all 27 cubelets.
4. **A missing verb does not stop a model, it makes one up.** With no way to clear the
   scene, it invented `import_asset("empty.glb")`.

There is also a cheap technique worth keeping: every builder's output is executed in
**headless Blender** before any model time is spent. That is what caught
`BLENDER_EEVEE_NEXT` not existing in Blender 5.2, and it costs seconds.

**Measured cost.** 25-60 s per round warm, so a 12-round run is 8-12 minutes. Overnight
agent, not interactive, as predicted.

**Not yet tested: `--reference`**, the inverse-graphics mode where the model builds until
its render matches a target image. That is the experiment that joins this half of the detour
to the other one.
