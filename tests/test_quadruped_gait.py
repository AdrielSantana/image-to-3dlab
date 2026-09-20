from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "scripts" / "quadruped_gait.py"


def load_module():
    spec = importlib.util.spec_from_file_location("quadruped_gait", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


G = load_module()
FOX = {"front_leg_length": 0.7052, "hind_leg_length": 0.6886, "body_height": 1.484}
RAM = {"front_leg_length": 0.7654, "hind_leg_length": 0.8288, "body_height": 1.557}


def test_walk_is_the_four_beat_lateral_sequence():
    """Mammals walk BL -> FL -> BR -> FR, evenly spaced."""
    order = sorted(G.LIMBS, key=lambda limb: G.limb_phase("walk", limb))
    assert order == ["BL", "FL", "BR", "FR"]
    spacing = [G.limb_phase("walk", limb) for limb in order]
    assert spacing == [0.0, 0.25, 0.5, 0.75]


def test_trot_pairs_the_diagonals():
    assert G.limb_phase("trot", "FL") == G.limb_phase("trot", "BR")
    assert G.limb_phase("trot", "FR") == G.limb_phase("trot", "BL")
    assert G.limb_phase("trot", "FL") != G.limb_phase("trot", "FR")


def test_a_planted_foot_never_leaves_the_ground():
    """The whole point of driving IK feet: zero lift through stance, or the foot skates."""
    for gait in G.GAITS:
        duty = G.GAITS[gait]["duty"]
        for limb in G.LIMBS:
            phase = G.limb_phase(gait, limb)
            for step in range(200):
                u = step / 200
                _, up = G.foot_offset(u, phase, duty, 0.2, 0.05)
                if G.is_planted(u, phase, duty):
                    assert up == 0.0, f"{gait}/{limb} lifted at u={u}"
                else:
                    assert up >= 0.0


def test_foot_cycle_is_continuous_across_the_stance_swing_seam():
    duty = 0.65
    before = G.foot_offset(duty - 1e-9, 0.0, duty, 0.2, 0.05)
    after = G.foot_offset(duty + 1e-9, 0.0, duty, 0.2, 0.05)
    assert before[0] == pytest.approx(after[0], abs=1e-6)
    assert before[1] == pytest.approx(after[1], abs=1e-6)


def test_the_cycle_loops_without_a_pop():
    """The last frame must repeat the first exactly, or the action visibly jumps."""
    for gait in G.GAITS:
        plan = G.gait_plan(32, gait, **FOX)
        for limb in G.LIMBS:
            first, last = plan["feet"][limb][0], plan["feet"][limb][-1]
            assert first["fore_aft"] == pytest.approx(last["fore_aft"], abs=1e-9)
            assert first["up"] == pytest.approx(last["up"], abs=1e-9)
        assert plan["body"][0]["up"] == pytest.approx(plan["body"][-1]["up"], abs=1e-9)


def test_exactly_the_right_feet_carry_weight():
    """A trot always has two feet down; a four-beat walk never has fewer than two."""
    for gait, expected in (("trot", {2}), ("walk", {2, 3})):
        duty = G.GAITS[gait]["duty"]
        seen = set()
        for step in range(400):
            u = step / 400
            down = sum(1 for limb in G.LIMBS
                       if G.is_planted(u, G.limb_phase(gait, limb), duty))
            seen.add(down)
        assert seen <= expected, f"{gait} had support counts {seen}"
        assert seen


def test_stride_and_lift_scale_with_the_rig_not_a_constant():
    """The ram and fox have inverted segments; absolutes must not be shared."""
    fox = G.stride_and_lift(FOX["front_leg_length"], FOX["hind_leg_length"])
    ram = G.stride_and_lift(RAM["front_leg_length"], RAM["hind_leg_length"])
    assert ram["hind_stride"] > fox["hind_stride"]
    assert ram["hind_stride"] / RAM["hind_leg_length"] == pytest.approx(
        fox["hind_stride"] / FOX["hind_leg_length"])


def test_constants_reproduce_the_accepted_ram_walk():
    """Guards the derivation: these came from RamWalk_Natural_Final, not from taste."""
    ram = G.stride_and_lift(RAM["front_leg_length"], RAM["hind_leg_length"])
    assert ram["hind_stride"] == pytest.approx(0.2050, abs=5e-4)
    assert ram["hind_lift"] == pytest.approx(0.0550, abs=5e-4)
    assert ram["front_stride"] == pytest.approx(0.1677, abs=5e-4)
    assert ram["front_lift"] == pytest.approx(0.0461, abs=5e-4)
    assert RAM["body_height"] * G.BOB_PER_HEIGHT == pytest.approx(0.0120, abs=5e-4)


def test_foot_lift_stays_near_three_percent_of_body_height():
    """The rejected generic cycle lifted 8.72% of body height; that is the bug."""
    for rig in (FOX, RAM):
        sizes = G.stride_and_lift(rig["front_leg_length"], rig["hind_leg_length"])
        assert 0.02 < sizes["front_lift"] / rig["body_height"] < 0.045


def test_the_body_actually_moves():
    """Zero body motion is precisely what made the generic cycle read as a marionette."""
    for gait in G.GAITS:
        plan = G.gait_plan(32, gait, **FOX)
        ups = [b["up"] for b in plan["body"]]
        assert max(ups) - min(ups) > 0.005


def test_walk_rolls_laterally_and_trot_does_not():
    walk = G.gait_plan(32, "walk", **FOX)
    trot = G.gait_plan(32, "trot", **FOX)
    assert max(abs(b["lateral"]) for b in walk["body"]) > 0
    assert max(abs(b["lateral"]) for b in trot["body"]) == 0


def test_scale_argument_scales_everything_together():
    plain = G.gait_plan(32, "walk", **FOX)
    big = G.gait_plan(32, "walk", **FOX, scale=2.0)
    assert big["sizes"]["hind_stride"] == pytest.approx(2 * plain["sizes"]["hind_stride"])
    assert max(b["up"] for b in big["body"]) == pytest.approx(
        2 * max(b["up"] for b in plain["body"]))


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5])
def test_duty_outside_the_open_unit_interval_is_refused(bad):
    with pytest.raises(ValueError):
        G.foot_offset(0.3, 0.0, bad, 0.2, 0.05)


def test_unknown_gait_and_limb_are_refused():
    with pytest.raises(ValueError, match="unknown gait"):
        G.limb_phase("gallop", "FL")
    with pytest.raises(ValueError, match="unknown limb"):
        G.limb_phase("walk", "XX")


def test_a_too_short_cycle_and_bad_leg_lengths_are_refused():
    with pytest.raises(ValueError):
        G.gait_plan(3, "walk", **FOX)
    with pytest.raises(ValueError):
        G.stride_and_lift(0.0, 0.5)


def test_control_names_match_the_rigify_basic_quadruped():
    assert set(G.FOOT_IK) == set(G.LIMBS)
    assert G.FOOT_IK["BL"] == "foot_ik.L"
    assert G.FOOT_IK["FL"] == "front_foot_ik.L"
    assert G.BODY_CHAIN == ("hips", "torso", "chest", "neck", "head")


def test_rest_extent_ignores_the_posed_bounding_box():
    """object.dimensions is pose-dependent: a fox caught mid-stride measured 1.675
    against a true 1.484, scaling every stride and bob 13% with it."""
    box = [(0, 0, 0), (0.744, 1.484, 1.404)]
    assert G.rest_extent(box) == pytest.approx(1.484)


def test_rest_extent_refuses_an_empty_mesh():
    with pytest.raises(ValueError):
        G.rest_extent([])


# --- the audit: every case below is a cycle this pipeline actually produced ------------

def good_walk():
    """The accepted fox walk: shoulder-absorbed reach, IK feet, live body."""
    return {
        "body_bob_pct_height": 0.0114 / 1.484,
        "front_rake_excursion_deg": 6.2,
        "loop_gap": 0.0,
        "limbs": {
            "FL": {"lift_pct_height": 0.0422 / 1.484, "stride_pct_leg": 0.1517 / 0.7052,
                   "planted_fraction": 22 / 33},
            "BL": {"lift_pct_height": 0.0454 / 1.484, "stride_pct_leg": 0.1672 / 0.6886,
                   "planted_fraction": 23 / 33},
        },
    }


def test_the_accepted_walk_passes_its_own_audit():
    assert G.check_gait(good_walk()) == []


def test_the_marionette_is_caught():
    """RigifyWalk left torso, hips and chest at exactly zero motion."""
    bad = good_walk()
    bad["body_bob_pct_height"] = 0.0
    problems = G.check_gait(bad)
    assert any("welded in place" in p for p in problems)


def test_the_high_stepping_cycle_is_caught():
    """The rejected cycle lifted 8.72% of body height against an accepted 2.96%."""
    bad = good_walk()
    bad["limbs"]["FL"]["lift_pct_height"] = 0.0872
    assert any("lifts 0.0872" in p for p in G.check_gait(bad))


def test_the_over_striding_cycle_is_caught():
    """RigifyWalk swung the front foot 0.5548 on a 0.7654 foreleg."""
    bad = good_walk()
    bad["limbs"]["FL"]["stride_pct_leg"] = 0.5548 / 0.7654
    assert any("strides" in p for p in G.check_gait(bad))


def test_skating_fk_forelegs_are_caught():
    """FK forelegs with no ground constraint held contact just 6 frames of 33."""
    bad = good_walk()
    bad["limbs"]["FL"]["planted_fraction"] = 6 / 33
    assert any("skates" in p for p in G.check_gait(bad))


def test_foreleg_rake_is_reported_but_not_gated():
    """The rake metric disagreed with itself (6.2 vs 10.6 on one cycle, depending on
    whether the authoring or evaluated bones were read), so it carries no threshold yet.
    This test pins that decision so nobody re-enables the gate without fixing the
    measurement first."""
    assert G.GAIT_LIMITS["front_rake_excursion_deg"] is None
    bad = good_walk()
    bad["front_rake_excursion_deg"] = 14.4
    assert not any("ankle is leading" in p for p in G.check_gait(bad))


def test_a_popping_loop_is_caught():
    bad = good_walk()
    bad["loop_gap"] = 0.01
    assert any("does not close" in p for p in G.check_gait(bad))


def test_the_audit_reports_every_independent_problem_at_once():
    """A bad run usually fails several ways; reporting one at a time wastes a cycle."""
    bad = good_walk()
    bad["body_bob_pct_height"] = 0.0
    bad["limbs"]["FL"]["planted_fraction"] = 6 / 33
    bad["limbs"]["FL"]["lift_pct_height"] = 0.0872
    assert len(G.check_gait(bad)) >= 3


def test_shoulder_rotation_is_off_by_default_because_it_broke_the_foreleg():
    """Rotating the shoulder 0.08 rad with nothing else animated hyperextended the elbow
    from 141 to 180 degrees and folded the wrist from 174 to 116 -- it caused the very
    ankle-forward defect it was added to fix. Pinned so it is not re-enabled by accident."""
    assert G.SHOULDER_SHARE == 0.0
    assert G.shoulder_angle(0.077, 0.7052) == 0.0


def test_shoulder_share_still_scales_when_asked_for_explicitly():
    assert G.shoulder_angle(0.077, 0.7052, 1.5) > G.shoulder_angle(0.077, 0.7052, 0.5)
    assert G.shoulder_angle(-0.077, 0.7052, 1.0) < 0
    with pytest.raises(ValueError):
        G.shoulder_angle(0.077, 0.0, 1.0)


# --- IK reach: the foreleg lock that read as an ankle shoved forward -------------------

FOX_FORE = {"chain_length": 0.5170, "vertical_drop": 0.4761, "horizontal_at_rest": 0.1049}
FOX_HIND = {"chain_length": 0.4431, "vertical_drop": 0.3532, "horizontal_at_rest": 0.1088}


def test_the_foreleg_headroom_that_caused_the_lock_is_measured():
    """The fox's foreleg rests at 94.3% extension with ~0.016 of travel each way; the
    cycle asked for 0.077 and the elbow locked at 180 degrees."""
    assert G.max_half_stride(**FOX_FORE) == pytest.approx(0.0158, abs=5e-4)
    assert G.max_half_stride(**FOX_HIND) == pytest.approx(0.1202, abs=5e-4)


def test_the_hind_leg_needs_no_drop_because_it_never_locked():
    assert G.body_drop_for_reach(**FOX_HIND, half_stride=0.0852) == 0.0


def test_the_foreleg_drop_buys_back_exactly_the_reach_it_needs():
    drop = G.body_drop_for_reach(**FOX_FORE, half_stride=0.0772)
    assert drop > 0
    lowered = dict(FOX_FORE, vertical_drop=FOX_FORE["vertical_drop"] - drop)
    assert G.max_half_stride(**lowered) == pytest.approx(0.0772, abs=1e-4)


def test_an_impossible_stride_is_refused_rather_than_silently_locked():
    with pytest.raises(ValueError, match="unreachable"):
        G.body_drop_for_reach(**FOX_FORE, half_stride=0.60)


# --- secondary motion: head and tail ---------------------------------------------------

def test_the_tail_whips_rather_than_swinging_as_one_board():
    """Each segment lags the one before it; a tail with no lag reads as a stiff plank."""
    base = G.tail_angles(0.3, "walk")
    assert len(base) == len(G.TAIL_CHAIN)
    assert len({round(seg["swing"], 6) for seg in base}) == len(base)


def test_tail_and_head_motion_loops_with_the_cycle():
    for gait in G.GAITS:
        for u in (0.0, 0.25, 0.5):
            assert G.tail_angles(u, gait)[0]["swing"] == pytest.approx(
                G.tail_angles(u + 1.0, gait)[0]["swing"], abs=1e-9)
            assert G.head_angles(u, gait)["head"]["nod"] == pytest.approx(
                G.head_angles(u + 1.0, gait)["head"]["nod"], abs=1e-9)


def test_the_head_counter_nods_against_the_body():
    """An animal keeps its head level while the body bobs; matching the bob looks robotic."""
    walk = G.gait_plan(32, "walk", **FOX)
    bobs = [b["up"] for b in walk["body"]]
    nods = [h["head"]["nod"] for h in walk["head"]]
    peak_bob = bobs.index(max(bobs))
    assert nods[peak_bob] < 0


def test_head_and_tail_actually_move():
    """The whole point: a rigid head and tail read as a puppet even with perfect legs.
    Measured at the tail base, since the chain tapers toward the tip."""
    walk = G.gait_plan(32, "walk", **FOX)
    base = [t["segments"][0]["swing"] for t in walk["tail"]]
    assert max(base) - min(base) > math.radians(2)
    yaw = [h["head"]["yaw"] for h in walk["head"]]
    assert max(yaw) - min(yaw) > math.radians(1)


def test_secondary_amplitudes_stay_subtle():
    """Secondary motion that competes with the gait is worse than none."""
    walk = G.gait_plan(32, "walk", **FOX)
    for entry in walk["tail"]:
        for seg in entry["segments"]:
            assert abs(seg["swing"]) <= math.radians(G.TAIL_SWING_DEG) + 1e-9
    for entry in walk["head"]:
        assert abs(entry["head"]["nod"]) <= math.radians(G.HEAD_NOD_DEG) + 1e-9


def test_body_roll_stays_well_under_the_vertical_bob():
    """Roll as large as the bob, with the tail swinging at the same stride frequency,
    reinforces into a disco strut -- the rejected 'Bee Gees walk'."""
    walk = G.gait_plan(32, "walk", **FOX)
    roll = max(abs(b["lateral"]) for b in walk["body"])
    bob = max(abs(b["up"]) for b in walk["body"])
    assert 0 < roll < bob * 0.5


def test_tail_swing_is_tunable_and_defaults_modest():
    assert G.TAIL_SWING_DEG <= 4.0
    loud = G.gait_plan(32, "walk", **FOX, tail_swing_deg=12.0)
    quiet = G.gait_plan(32, "walk", **FOX)
    def peak(plan):
        return max(abs(seg["swing"]) for e in plan["tail"] for seg in e["segments"])

    assert peak(loud) > peak(quiet)


def test_the_tail_tip_does_not_trace_a_circle():
    """Swing and lift must share a frequency. Driving them at different rates makes the
    tip orbit -- observed as the tail 'moving in a circular motion' rather than trailing."""
    assert G.TAIL_LIFT_DEG == 0.0
    loud = G.tail_angles(0.0, "walk", lift_deg=5.0)
    quarter = G.tail_angles(0.25, "walk", lift_deg=5.0)
    # same frequency => swing and lift stay in fixed proportion, so no orbit
    for a, b in zip(loud, quarter):
        if abs(a["swing"]) > 1e-9 and abs(b["swing"]) > 1e-9:
            assert a["lift"] / a["swing"] == pytest.approx(b["lift"] / b["swing"], abs=1e-6)


def test_tail_segments_taper_so_the_tip_does_not_accumulate():
    """Four segments each turning the full amount put four times that at the tip.

    Compares peak amplitude across the cycle, not one instant: the segments are
    phase-lagged, so at any single moment the later ones can read larger.
    """
    peaks = []
    for index in range(len(G.TAIL_CHAIN)):
        values = [abs(G.tail_angles(step / 64, "walk")[index]["swing"])
                  for step in range(64)]
        peaks.append(max(values))
    assert peaks == sorted(peaks, reverse=True)
    assert peaks[-1] < peaks[0] * 0.5


def test_lift_tracks_stride_so_the_gait_cannot_march():
    """Lift approaching stride means the foot rises as far as it travels -- a parade
    step. Observed at 1.18 when lift was set independently of stride."""
    for gait in G.GAITS:
        plan = G.gait_plan(24, gait, **FOX)
        for limb in ("front", "hind"):
            ratio = plan["sizes"][f"{limb}_lift"] / plan["sizes"][f"{limb}_stride"]
            assert 0.15 < ratio < 0.45, f"{gait}/{limb} lift:stride is {ratio:.2f}"


def test_growing_the_stride_grows_the_lift_with_it():
    small = G.lift_from_stride(0.15, 0.17)
    big = G.lift_from_stride(0.30, 0.34)
    assert big["front_lift"] == pytest.approx(2 * small["front_lift"])
    assert (big["front_lift"] / big["front_stride"]
            == pytest.approx(small["front_lift"] / small["front_stride"]))
