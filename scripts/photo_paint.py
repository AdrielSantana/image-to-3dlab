#!/usr/bin/env python3
"""Paint a finished model with the real pixels of its source photos, where they can see.

    python scripts/photo_paint.py finished.glb views_dir out.glb [--weights weights.png]

`views_dir` holds a `transforms.json` and the view images, as pixal3d.cpp stages next to
every run (`<run>.svviews/`). Every surface a view can see, facing it and away from the
silhouette edge, takes that photo's pixel; everything else keeps the model's own paint.
That is what brings back text, logos and faces that a repaint redraws as lookalikes.

The GLB must be one mesh with one base-colour texture and no node transforms, which is
what Finish writes. Only the base-colour image changes; every other byte stays.

`--weights` writes the per-texel trust map (white = photo, black = own paint), the first
thing to look at when a result looks wrong.

The logic lives in `image_to_3dlab/photo_paint.py`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image

from image_to_3dlab import photo_paint as pp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="finished GLB (Finish's output)")
    parser.add_argument("views", type=Path, help="directory with transforms.json + images")
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, help="also write the trust map as a PNG")
    defaults = pp.Settings()
    parser.add_argument("--facing-low", type=float, default=defaults.facing_low)
    parser.add_argument("--facing-high", type=float, default=defaults.facing_high)
    parser.add_argument("--edge-px", type=int, default=defaults.edge_px)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for path in (args.model, args.views / "transforms.json"):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")
    started = time.time()
    positions, uvs, faces, texture = pp.read_glb(args.model)
    views, mesh_scale = pp.load_views(args.views)
    settings = pp.Settings(facing_low=args.facing_low, facing_high=args.facing_high,
                           edge_px=args.edge_px)
    painted, weight = pp.paint_texture(texture, positions, uvs, faces, views, mesh_scale,
                                       settings=settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pp.replace_base_colour(args.model.read_bytes(),
                                                   pp.encode_png(painted)))
    if args.weights:
        Image.fromarray(np.rint(weight * 255).astype(np.uint8)).save(args.weights)
    share = float((weight > 0.5).mean())
    print(f"[photo-paint] {len(views)} view(s), {share:.0%} of the texture from the photo, "
          f"{time.time() - started:.0f}s -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
