"""SF3D picks the fastest device the machine has."""

from __future__ import annotations

import pytest

from image_to_3dlab.sf3d_backend import pick_device


@pytest.mark.parametrize("forced_cpu,cuda,mps,expected", [
    (False, True, False, "cuda"),
    (False, False, True, "mps"),
    (False, False, False, "cpu"),
    (True, True, False, "cpu"),
    (True, False, True, "cpu"),
])
def test_pick_device(forced_cpu, cuda, mps, expected):
    assert pick_device(forced_cpu, cuda, mps) == expected
