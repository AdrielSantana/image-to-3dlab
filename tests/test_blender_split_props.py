"""Tests for splitting a prop sheet into upright, named props.

The bpy-free parts decide everything that is hard to see in a render: which loose parts
are one prop, which prop gets which name, and how far each one is turned. A wrong grouping
welds two props into one; a wrong order names the crate "barrel"; a wrong turn stands a
prop on its side, or swings a round prop's painted front away from the camera.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_split_props.py"


def _load():
    spec = importlib.util.spec_from_file_location("split_props", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = _load()


def box_points(width: float, depth: float, height: float, count: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.uniform(-0.5, 0.5, (count, 3)) * np.array([width, depth, height])


def disc_points(radius: float, count: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(2)
    angle = rng.uniform(0, 2 * np.pi, count)
    r = radius * np.sqrt(rng.uniform(0, 1, count))
    return np.stack([r * np.cos(angle), r * np.sin(angle)], axis=1)


def test_arguments_read_names_turns_and_defaults():
    args = split.parse_args(["blender", "--", "in.glb", "out", "--names", "barrel", "crate",
                             "--turn", "chest=90", "--turn", "crate=-12.5"])
    assert args.source == Path("in.glb")
    assert args.out_dir == Path("out")
    assert args.names == ["barrel", "crate"]
    assert args.turn == {"chest": 90.0, "crate": -12.5}
    assert args.straighten is True
    assert args.yaw_threshold == split.YAW_THRESHOLD


def test_a_turn_without_degrees_is_refused():
    with pytest.raises(SystemExit):
        split.parse_args(["--", "in.glb", "out", "--turn", "chest"])
    with pytest.raises(SystemExit):
        split.parse_args(["--", "in.glb", "out", "--turn", "chest=a lot"])


def test_a_yaw_threshold_outside_a_fraction_is_refused():
    with pytest.raises(SystemExit):
        split.parse_args(["--", "in.glb", "out", "--yaw-threshold", "10"])


def test_boxes_touch_in_the_front_view_whatever_their_depth():
    front = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    behind = ((0.5, 5.0, 0.5), (1.5, 6.0, 1.5))
    beside = ((1.2, 0.0, 0.0), (2.0, 1.0, 1.0))
    assert split.boxes_touch(front, behind, margin=0.0)
    assert not split.boxes_touch(front, beside, margin=0.0)
    assert split.boxes_touch(front, beside, margin=0.25)


def test_a_lid_joins_its_chest_and_the_neighbour_stays_apart():
    chest = ((0.0, 0.0, 0.0), (1.0, 1.0, 0.7))
    lid = ((0.0, 0.1, 0.65), (1.0, 0.9, 1.0))
    neighbour = ((1.5, 0.0, 0.0), (2.5, 1.0, 1.0))
    assert split.group_touching([chest, neighbour, lid], margin=0.01) == [[0, 2], [1]]


def test_touching_chains_become_one_prop():
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((0.9, 0.0, 0.0), (2.0, 1.0, 1.0))
    c = ((1.9, 0.0, 0.0), (3.0, 1.0, 1.0))
    assert split.group_touching([a, c, b], margin=0.0) == [[0, 1, 2]]


def test_crumbs_are_dropped_and_props_kept():
    assert split.keep_large([1000, 5, 900, 12], share=0.01) == [0, 2]


def test_reading_order_is_top_row_first_and_left_is_plus_x():
    labels, centres = [], []
    for row, z in enumerate((2.0, 1.0, 0.0)):
        for column, x in enumerate((1.0, 0.0, -1.0)):   # +X is the viewer's left
            labels.append(f"r{row}c{column}")
            centres.append((x + random.Random(row * 3 + column).uniform(-0.1, 0.1),
                            z + random.Random(9 - row * 3 - column).uniform(-0.15, 0.15)))
    shuffled = list(range(9))
    random.Random(7).shuffle(shuffled)
    order = split.reading_order([centres[i] for i in shuffled], [0.8] * 9)
    assert [labels[shuffled[i]] for i in order] == labels


def test_an_upright_box_is_left_upright():
    tx, ty = split.best_tilt(box_points(0.2, 0.15, 0.25))
    assert abs(tx) <= 0.3 and abs(ty) <= 0.3


@pytest.mark.parametrize(("tilt_x", "tilt_y"), [(28.0, 0.0), (-17.0, 0.0), (0.0, 6.0)])
def test_a_tipped_box_is_stood_back_up(tilt_x, tilt_y):
    points = box_points(0.2, 0.15, 0.25) @ split.tilt_matrix(tilt_x, tilt_y).T
    tx, ty = split.best_tilt(points)
    assert tx == pytest.approx(-tilt_x, abs=0.3)
    assert ty == pytest.approx(-tilt_y, abs=0.3)


def test_a_turned_box_is_squared_up():
    footprint = box_points(0.3, 0.1, 0.1)[:, :2]
    degrees, gain = split.best_yaw(split.rotate_xy(footprint, 30.0))
    assert degrees == pytest.approx(-30.0, abs=0.3)
    assert gain > split.YAW_THRESHOLD


def test_a_box_drawn_corner_on_is_turned_the_short_way():
    footprint = box_points(0.2, 0.2, 0.2)[:, :2]
    degrees, _ = split.best_yaw(split.rotate_xy(footprint, 41.5))
    assert degrees == pytest.approx(-41.5, abs=0.3)


def test_a_round_prop_gains_too_little_to_be_turned():
    _, gain = split.best_yaw(disc_points(0.1))
    assert gain < split.YAW_THRESHOLD


def test_turns_near_45_degrees_are_flagged_as_ties():
    assert split.yaw_is_tie(43.75)
    assert split.yaw_is_tie(-41.5)
    assert not split.yaw_is_tie(34.25)
    assert not split.yaw_is_tie(0.0)


def test_props_without_a_name_are_numbered():
    assert split.default_name(0) == "prop_01"
    assert split.default_name(11) == "prop_12"
