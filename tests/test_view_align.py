"""Tests for image_to_3dlab.view_align: nudging a picture onto the model before painting."""

from __future__ import annotations

import numpy as np
import pytest

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import view_align as va

IDENTITY = ((0, 1, 2), (1, 1, 1))


def _disc(size: int, radius: float, centre=(0.0, 0.0)) -> np.ndarray:
    ys, xs = np.mgrid[:size, :size] + 0.5
    cy, cx = size / 2 + centre[0], size / 2 + centre[1]
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2


def _texture_image(size: int, seed: int = 0) -> np.ndarray:
    """Blobby grey texture: features at a few-pixel scale, unique everywhere."""
    from scipy.ndimage import gaussian_filter
    noise = np.random.default_rng(seed).random((size, size))
    image = gaussian_filter(noise, 2.0)
    image = (image - image.min()) / np.ptp(image) * 255
    return np.repeat(image[..., None], 3, axis=2).astype(np.uint8)


# --- outline fit ----------------------------------------------------------------------------


def test_outline_fit_recovers_a_known_scale_and_shift():
    model = _disc(256, 80)
    picture = _disc(256, 80 * 1.02, centre=(-3, 4))
    fit = va.fit_outline(model, picture, max_shift=6)
    assert fit.scale == pytest.approx(1.02, abs=0.006)
    assert (fit.dy, fit.dx) == (pytest.approx(-3, abs=1), pytest.approx(4, abs=1))
    assert fit.iou_after > fit.iou_before


def test_outline_fit_leaves_a_matching_outline_alone():
    model = _disc(256, 80)
    fit = va.fit_outline(model, model.copy(), max_shift=6)
    assert (fit.scale, fit.dx, fit.dy) == (1.0, 0, 0)


def test_outline_fit_does_nothing_when_the_camera_sees_no_model():
    fit = va.fit_outline(np.zeros((64, 64), bool), _disc(64, 20), max_shift=6)
    assert (fit.scale, fit.dx, fit.dy) == (1.0, 0, 0)


# --- feature matching ----------------------------------------------------------------------


def test_feature_matching_finds_a_small_shift_in_textured_areas():
    own = _texture_image(256)
    picture = np.roll(own, (2, -3), axis=(0, 1))  # picture = own moved down 2, left 3
    mask = np.ones((256, 256), dtype=bool)
    field, matches = va.match_features(own, picture, mask, max_px=6)
    assert matches > 20
    inner = field[40:-40, 40:-40]
    # own(p) ~ picture(p + field): field points from the model to the picture.
    assert np.median(inner[..., 1]) == pytest.approx(2, abs=0.5)
    assert np.median(inner[..., 0]) == pytest.approx(-3, abs=0.5)


def test_feature_matching_leaves_flat_areas_alone():
    flat = np.full((256, 256, 3), 120, dtype=np.uint8)
    field, matches = va.match_features(flat, flat.copy(), np.ones((256, 256), bool), max_px=6)
    assert matches == 0 and not field.any()


def test_feature_matching_ignores_repeating_patterns():
    # Stripes every 4 px: several equally good matches within reach, so none is trusted.
    stripes = np.zeros((256, 256, 3), dtype=np.uint8)
    stripes[:, ::4] = 255
    stripes[:, 1::4] = 255
    field, matches = va.match_features(stripes, np.roll(stripes, 1, axis=1),
                                       np.ones((256, 256), bool), max_px=6)
    assert matches == 0 and not field.any()


def test_feature_matching_never_moves_more_than_the_cap():
    own = _texture_image(256, seed=1)
    picture = np.roll(own, 5, axis=0)
    field, _ = va.match_features(own, picture, np.ones((256, 256), bool), max_px=3)
    assert np.linalg.norm(field, axis=2).max() <= 3 + 1e-6


# --- whole view ----------------------------------------------------------------------------


def _quad():
    positions = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]],
                         dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)
    return positions, faces, uvs


def _camera(image: np.ndarray) -> pp.View:
    c2w = np.eye(4)
    c2w[2, 3] = 3.0
    return pp.View(image, c2w, 0.6, "front")


def test_align_view_puts_a_shifted_picture_back_on_the_model():
    positions, faces, uvs = _quad()
    size = 384
    own_tex = _texture_image(256, seed=3)
    view = _camera(np.zeros((size, size, 4), dtype=np.uint8))
    # The model's own paint as the camera sees it, used as a perfectly aligned "picture"...
    seen, rendered = va.render_own_paint(positions, uvs, faces, own_tex, view, IDENTITY, 1.0)
    aligned = np.zeros((size, size, 4), dtype=np.uint8)
    aligned[..., :3] = rendered
    aligned[..., 3] = seen * 255
    # ...then knocked 3 px right and 2 px down, as a slightly-off camera would.
    shifted = np.roll(aligned, (2, 3), axis=(0, 1))
    fixed, report = va.align_view(_camera(shifted), positions, uvs, faces, own_tex, IDENTITY, 1.0)
    inner = (slice(size // 4, -size // 4),) * 2
    before = np.abs(shifted[inner][..., :3].astype(int) - aligned[inner][..., :3]).mean()
    after = np.abs(fixed.image[inner][..., :3].astype(int) - aligned[inner][..., :3]).mean()
    assert report.iou_after > report.iou_before
    assert after < before * 0.4
    assert np.allclose(fixed.cam_to_world, _camera(shifted).cam_to_world)
