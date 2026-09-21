# Resume guidance

Where to pick up. Updated at the end of every session; kept to a handful of bullets on
purpose, because the detail lives in `CHANGELOG.md` and the dated `docs/*.md`.

Session notes that are not part of the shipped repo live in `journal/` (git-ignored).
Start there too if a bullet below points at one.

**2026-09-21**

- **IN PROGRESS: Setup & Status page.** `AGENTS.md` now forbids downloading weights before
  the user has confirmed which pipeline and which route. `viewer/backend_catalog.py` plus
  `GET /api/catalog` describe what each backend costs, and `viewer/download_api.py` runs a
  download with measured progress (directory growth against the expected size, not parsed
  tqdm), stall detection and cancel. The page itself is half built: it lands first, has a
  "skip this next time" checkbox, and Generate's old Setup card is now a one-line health
  strip. **Owed:** the three unconsented paths still exist. `bootstrap_pixal3d_cpp.sh`
  pulls 8.1 GB unconditionally and `download_weights.py` defaulted to all three shape
  models (now pinned to 2.0 when driven from the viewer, still unpinned on the CLI).
- **Disk: 104 GB of Hugging Face cache went to 47 GB.** A rejected PyTorch Pixal3D port was
  still cached in full, and `download_shape` kept both the `.ckpt` it downloaded and the
  `.safetensors` it converted. The converter cleans up after itself now, and
  `scripts/audit_model_weights.py` finds the rest (dry run by default; never deletes
  anything it cannot prove is duplicated).
- **The viewer's Finish mode was silently losing finished assets.** The worker's own
  `I2L_STAGE::done` reached the browser as the job's completion event without any artifact
  URLs, so the page closed its stream before the real one arrived. Fixed, along with the
  progress panel never ticking its last stage and the Finish progress bar never filling.
  The repaint's own `step 7/15` output now drives real per-stage progress and ETAs.
- **README is visual now**, opening on three source images above the models generated from
  them, plus a 360° turntable. Built from renders the promo reel already produced; see
  `docs/images/README.md` for the size budget before adding more.
- **Still open from 2026-09-20:** multi-view conditioning is the real lever for the warrior
  girl's blade direction (`--fov` was tested and is *not* the cause); the pipeline still
  needs running across assets that fail differently (hard-surface, fur, a face); the decode
  crash at 1024 (`journal/decode-crash-debug.md`); and `--gss` defaults to 10 in
  `pixal3d_generate.py` now, which is correct.
- **NEXT, big release: make the repo CUDA-compliant as well as Apple Silicon**, so NVIDIA
  users can run it. Committed, not "later". Test on RunPod, run every pipeline, write
  NVIDIA onboarding. `pixal3d.cpp` already builds for CUDA and Vulkan, so it certifies
  first.
- **Parked, endorsed, not started: a promo/comms tool.** Game developers can build and
  cannot market, so producing something shareable straight out of the workflow is a real
  feature rather than a nicety. `scripts/build_showcase_reel.py` and the
  `blender_turntable.py` flags are written and deliberately uncommitted; see the
  `promo-video-tool-lane` note. Two WoW characters have been taken image → 3D and are
  wanted as a showcase reel like the earlier promos.
