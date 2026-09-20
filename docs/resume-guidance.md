# Resume guidance

Where to pick up. Updated at the end of every session; kept to a handful of bullets on
purpose — the detail lives in `CHANGELOG.md` and the dated `docs/*.md`.

**2026-09-20**

- **Pixal3D won the bake-off** and is the route now: one pass, ~6 min, saturated colour, no
  repaint stage. Run it with `scripts/pixal3d_generate.py`, or from the viewer. Full numbers
  in `docs/pixal3d-evaluation-2026-09-20.md`. Use `raven38/pixal3d.cpp` (C++/GGML), never the
  PyTorch Mac port — it needs 22 GB resident and will not fit 32 GB.
- **Raise `--gss` to 10.** At the 7.5 default the warrior girl lost her sword blade entirely;
  at 10 it came back and her proportions tightened. 13 is worse — the sword detaches from the
  hand. The viewer still ships 7.5 as the default and should not.
- **NEXT: multi-view.** The blade at gss 10 points toward her front instead of along her arm.
  Probably depth ambiguity, which 4-view conditioning is built to fix. Needs the MV weight set
  (`raven38/pixal3d-q8_0-v1`, ~8 GB) and four consistent turnarounds — generating those
  consistently is the real problem, not the plumbing. The cheap alternative was tested and
  **failed**: `--fov 0.7235` (41°, MoGe-like) against the 20° gauge default left the blade
  pointing the same wrong way, so the gauge camera is not the cause and multi-view is the
  real lever.
- **Owed: run the pipeline across more assets.** Every quality conclusion so far rests on the
  fox, the Snag and the warrior girl. Pick subjects that fail differently — hard-surface, fur,
  a face.
- **LATER, big release: make the repo CUDA-compliant as well as Apple Silicon**, so NVIDIA
  users can run it. Test on RunPod (the MCP tools are already wired up here), run every
  pipeline, and write NVIDIA onboarding. `pixal3d.cpp` already builds for CUDA and Vulkan,
  which makes it the natural first backend to certify.
- **Open from earlier lanes:** the decode crash at 1024 (`journal/decode-crash-debug.md`),
  and normal-map baking works but gains little on assets whose retopology already tracks the
  decode closely.
