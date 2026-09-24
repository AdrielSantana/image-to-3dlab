"""Is this image already cut out? Judged by its contents, never by its mode.

Qwen-Image through `stable-diffusion.cpp` writes RGBA whose alpha is opaque noise with
nothing transparent in it. Backends that read "has an alpha channel" as "already matted"
skip background removal, and the backdrop comes back as geometry: sheets beside a fox's
head in Pixal3D (2026-09-22), a grey slab around every SF3D model (2026-09-24).
"""

from __future__ import annotations

# A real cutout leaves a lot of the frame empty -- a centred subject is typically 30-60%
# transparent. This floor only has to separate that from an alpha channel that cuts nothing.
MATTE_MIN_TRANSPARENT = 0.02
MATTE_TRANSPARENT_BELOW = 16


def is_matted(image) -> bool:
    """True only when a meaningful share of the PIL image is actually transparent."""
    if image.mode not in ("RGBA", "LA") and "transparency" not in image.info:
        return False
    alpha = image.convert("RGBA").getchannel("A")
    transparent = sum(alpha.histogram()[:MATTE_TRANSPARENT_BELOW])
    return transparent / (alpha.width * alpha.height) >= MATTE_MIN_TRANSPARENT
