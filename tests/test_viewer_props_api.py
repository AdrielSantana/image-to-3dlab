"""Tests for the browser prop-sheet job.

Everything a browser sends becomes a Blender or a script argument, so the settings tests
cover what a page can send: bad names, LODs in the wrong order, stray keys. The rest
covers the two things a user sees go wrong: a progress bar that stalls or goes backwards,
and a result list that loses props after a turn re-bakes one of them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "viewer" / "props_api.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


props = _load("props_api", MODULE)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_defaults_are_used_when_nothing_is_sent():
    settings = props.normalise_settings({})
    assert settings == props.DEFAULT_SETTINGS
    assert settings["lods"] is not props.DEFAULT_SETTINGS["lods"]  # a copy


def test_the_yaw_threshold_default_is_the_splitters():
    """Repeated here because the splitter imports numpy and the server must not."""
    sys.path.insert(0, str(ROOT / "scripts"))
    splitter = _load("split_props_for_api", ROOT / "scripts" / "blender_split_props.py")
    assert props.YAW_THRESHOLD == splitter.YAW_THRESHOLD


def test_the_lod_defaults_are_the_finishers():
    finisher = props.finisher_module()
    assert props.DEFAULT_SETTINGS["lods"] == list(finisher.DEFAULT_LODS)
    assert props.DEFAULT_SETTINGS["atlas"] == finisher.DEFAULT_ATLAS
    assert props.ATLAS_SIZES == finisher.ATLAS_SIZES
    assert props.FACE_RANGE == finisher.FACE_RANGE


def test_names_arrive_one_per_line_from_the_textarea():
    settings = props.normalise_settings({"names": "barrel\n  crate \n\nchest\n"})
    assert settings["names"] == ["barrel", "crate", "chest"]


@pytest.mark.parametrize("names", [
    "barrel\nbarrel", "iron chest", "../etc", "a" * 41, ["x"] * 65, 12,
])
def test_names_that_cannot_be_file_names_or_repeat_are_refused(names):
    with pytest.raises(ValueError) as caught:
        props.normalise_settings({"names": names})
    assert "names" in str(caught.value)


def test_lods_arrive_as_text_or_a_list():
    assert props.normalise_settings({"lods": "8000,3000"})["lods"] == [8000, 3000]
    assert props.normalise_settings({"lods": [4000]})["lods"] == [4000]


@pytest.mark.parametrize("lods", [
    "", "5000,500", "5000,250000", "1000,2500", "3000,3000", "9000,8000,7000,6000,5000", "lots",
])
def test_lods_the_finisher_would_refuse_are_refused_first(lods):
    with pytest.raises(ValueError) as caught:
        props.normalise_settings({"lods": lods})
    assert "lods" in str(caught.value)


@pytest.mark.parametrize(("key", "value"), [
    ("atlas", 3000), ("atlas", "big"), ("metallic", 1.5), ("roughness", -0.1),
    ("ior", 0.5), ("yaw_threshold", 1.0), ("metallic", True),
])
def test_out_of_range_values_are_refused_with_their_key(key, value):
    with pytest.raises(ValueError) as caught:
        props.normalise_settings({key: value})
    assert key in str(caught.value)


def test_unknown_keys_are_dropped_rather_than_forwarded():
    settings = props.normalise_settings({"lodz": "1", "metallic": 0.4})
    assert "lodz" not in settings
    assert settings["metallic"] == pytest.approx(0.4)


@pytest.mark.parametrize(("degrees", "wrapped"), [
    (90, 90), (270, -90), (360, 0), (-180, 180), (180, 180), (450, 90),
])
def test_turns_wrap_so_four_quarter_turns_are_none(degrees, wrapped):
    assert props.wrap_degrees(degrees) == wrapped


def _job(tmp_path: Path, **settings):
    job = props.PropsJob("0" * 32, tmp_path)
    job.settings = props.normalise_settings(settings)
    return job


def test_the_split_command_carries_names_turns_and_threshold(tmp_path):
    job = _job(tmp_path, names="barrel\nchest", turns={"chest": 90})
    command = props.split_command(job, Path("/bin/blender"))
    assert command[:6] == ["/bin/blender", "-b", "--factory-startup", "-P",
                           str(props.SPLITTER), "--"]
    assert command[6:8] == [str(job.source_glb), str(job.split_dir)]
    assert command[command.index("--names") + 1:][:2] == ["barrel", "chest"]
    assert command[command.index("--turn") + 1] == "chest=90"
    assert command[command.index("--yaw-threshold") + 1] == str(props.YAW_THRESHOLD)


def test_the_split_command_numbers_props_when_no_names_are_given(tmp_path):
    assert "--names" not in props.split_command(_job(tmp_path), Path("/bin/blender"))


def test_the_split_command_is_one_the_splitter_accepts(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    splitter = _load("split_props_for_cmd", ROOT / "scripts" / "blender_split_props.py")
    job = _job(tmp_path, names="barrel\nchest", turns={"chest": -90})
    args = splitter.parse_args(props.split_command(job, Path("/bin/blender")))
    assert args.names == ["barrel", "chest"]
    assert args.turn == {"chest": -90.0}


def test_the_finish_command_is_one_the_finisher_accepts(tmp_path):
    job = _job(tmp_path, lods="6000,2000", atlas=2048)
    command = props.finish_command(job, Path("/bin/blender"), Path("/bin/gltfpack"))
    parsed = props.finisher_module().build_parser().parse_args(command[3:])
    assert parsed.source == job.split_dir
    assert parsed.out_dir == job.finished_dir
    assert parsed.lods == "6000,2000"
    assert parsed.atlas == 2048
    assert parsed.gltfpack == Path("/bin/gltfpack")
    assert parsed.blender == Path("/bin/blender")
    assert not parsed.no_compress


def test_a_turn_bakes_only_its_prop(tmp_path):
    command = props.finish_command(_job(tmp_path), Path("/bin/blender"), None, only="chest")
    assert command[3] == str(tmp_path / "split" / "chest.glb")


def test_without_gltfpack_or_when_asked_the_lods_stay_uncompressed(tmp_path):
    missing = props.finish_command(_job(tmp_path), Path("/bin/blender"), None)
    declined = props.finish_command(_job(tmp_path, compress=False), Path("/bin/blender"),
                                    Path("/bin/gltfpack"))
    for command in (missing, declined):
        assert "--no-compress" in command
        assert "--gltfpack" not in command


def test_rows_follow_the_order_the_finisher_bakes_in(tmp_path):
    """Not reading order and not a plain sort: `a-b.glb` sorts before `a.glb`."""
    for name in ("tree_stump", "anvil", "a", "a-b"):
        (tmp_path / f"{name}.glb").write_bytes(b"x")
    (tmp_path / "props.json").write_text("{}")
    order = props.bake_order(tmp_path, ["tree_stump", "anvil", "a", "a-b"])
    assert order == [p.stem for p in props.finisher_module().collect_props(tmp_path)]
    assert order == ["a-b", "a", "anvil", "tree_stump"]


def test_a_turn_bakes_its_one_prop_whatever_else_is_there(tmp_path):
    for name in ("anvil", "chest"):
        (tmp_path / f"{name}.glb").write_bytes(b"x")
    assert props.bake_order(tmp_path, ["chest"]) == ["chest"]


def test_prop_rows_cannot_collide_with_the_panels_own_phases():
    meta = props.stage_meta(["done", "split"])
    assert meta["stages"] == ["split", "prop:done", "prop:split"]
    assert meta["stage_labels"]["prop:done"] == "done"


# The lines `finish_props.py` prints, in order, for two props and two LODs.
FINISH_LINES = [
    "[barrel LOD0] /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup ...",
    "[barrel LOD0 gltfpack] /usr/bin/gltfpack -i a.glb -o b.glb ...",
    "[barrel LOD1] /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup ...",
    "[barrel LOD1 gltfpack] /usr/bin/gltfpack -i a.glb -o b.glb ...",
    "barrel: LOD0 425 KB, LOD1 387 KB",
    "[crate LOD0] /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup ...",
    "[crate LOD1] /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup ...",
    "crate: LOD0 239 KB, LOD1 227 KB",
    "finished 2 props in 30s -> out",
]


def test_progress_runs_forward_through_a_real_run():
    clock = Clock()
    progress = props.PropsProgress(clock=clock)
    seen = [progress.begin_split()["overall_pct"]]
    clock.now = 8
    seen.append(progress.split_line("SPLIT:: wrote 2 props to out")["overall_pct"])
    start = progress.begin_props(["barrel", "crate"], lods=2)
    assert start["stages"] == ["split", "prop:barrel", "prop:crate"]
    seen.append(start["overall_pct"])
    for line in FINISH_LINES:
        clock.now += 5
        event = progress.finish_line(line)
        if event:
            seen.append(event["overall_pct"])
    assert seen == sorted(seen)
    assert seen[0] == 0
    assert seen[-1] == 100


def test_each_prop_row_counts_its_lods_and_ticks_when_done():
    progress = props.PropsProgress(clock=Clock())
    progress.begin_props(["barrel", "crate"], lods=2)
    events = [progress.finish_line(line) for line in FINISH_LINES[:5]]
    assert events[0]["phase"] == "prop:barrel"
    assert (events[0]["step"], events[0]["total"]) == (0, 2)
    assert (events[2]["step"], events[2]["total"]) == (1, 2)
    assert events[4]["stage_pct"] == 100
    assert events[4]["overall_pct"] == pytest.approx(props.SPLIT_WEIGHT + (100 - props.SPLIT_WEIGHT) / 2)


def test_gltfpack_and_stray_lines_are_not_counted_as_bakes():
    progress = props.PropsProgress(clock=Clock())
    progress.begin_props(["barrel"], lods=3)
    assert progress.finish_line(FINISH_LINES[1]) is None
    assert progress.finish_line("[retopo] something else") is None
    assert progress.finish_line("[stranger LOD0] /bin/blender ...") is None


def test_split_output_other_than_its_own_markers_is_ignored():
    progress = props.PropsProgress(clock=Clock())
    assert progress.split_line("Blender 4.5.0 (hash abc)") is None
    assert progress.split_line("SPLIT:: 27 loose parts")["message"] == "27 loose parts"


def _write_run(root: Path, name: str = "sheet__props__20260925-120000",
               props_names=("barrel", "chest"), baked=2, web=True) -> Path:
    """A run with two LODs configured and `baked` of them on disk for every prop."""
    directory = root / name
    (directory / "split").mkdir(parents=True)
    (directory / "source.glb").write_bytes(b"glb")
    (directory / "settings.json").write_text(json.dumps(
        props.normalise_settings({"lods": [5000, 1000]})))
    (directory / "split" / "props.json").write_text(json.dumps({"props": [
        {"name": n, "faces": 1000, "size": [0.1, 0.1, 0.2], "yaw_degrees": 44.0 if n == "chest" else 0.0,
         "yaw_tie": n == "chest"} for n in props_names]}))
    for n in props_names:
        (directory / "split" / f"{n}.glb").write_bytes(b"prop")
        for index in range(baked):
            lod = directory / "finished" / n / f"{n}_LOD{index}.glb"
            lod.parent.mkdir(parents=True, exist_ok=True)
            lod.write_bytes(b"x" * 100)
            if web:
                lod.with_name(f"{n}_LOD{index}.web.glb").write_bytes(b"x" * 10)
    return directory


def test_a_run_is_described_from_disk_in_reading_order(tmp_path, monkeypatch):
    monkeypatch.setattr(props, "served_url", lambda p: "/" + p.relative_to(tmp_path).as_posix())
    run = props.describe_run(_write_run(tmp_path))
    assert [p["name"] for p in run["props"]] == ["barrel", "chest"]
    assert run["finished"]
    chest = run["props"][1]
    assert chest["yaw_tie"]
    assert [lod["bytes"] for lod in chest["lods"]] == [100, 100]
    assert chest["lods"][0]["web_bytes"] == 10
    # The plain LOD0: the viewer has no meshopt decoder for the .web one.
    assert chest["preview_url"].endswith("chest/chest_LOD0.glb")


def test_a_turn_rewriting_the_finish_record_does_not_hide_other_props(tmp_path):
    directory = _write_run(tmp_path)
    (directory / "finished" / "finish_props.json").write_text(json.dumps(
        {"props": [{"name": "chest"}]}))
    assert [p["name"] for p in props.describe_run(directory)["props"]] == ["barrel", "chest"]


def test_a_run_without_lods_yet_previews_the_split_prop(tmp_path, monkeypatch):
    monkeypatch.setattr(props, "served_url", lambda p: "/" + p.relative_to(tmp_path).as_posix())
    directory = _write_run(tmp_path, baked=0)
    run = props.describe_run(directory)
    assert not run["finished"]
    assert run["props"][0]["preview_url"].endswith("split/barrel.glb")


def test_runs_are_listed_and_strangers_ignored(tmp_path):
    _write_run(tmp_path)
    (tmp_path / "not-a-run").mkdir()
    runs = props.list_runs(tmp_path)
    assert [r["directory"] for r in runs] == ["sheet__props__20260925-120000"]


@pytest.mark.parametrize("name", ["../finish", "sheet__props__x", "", "sheet__finish__20260925-120000"])
def test_a_run_name_that_create_would_not_make_is_refused(tmp_path, name):
    with pytest.raises(RuntimeError):
        props.run_directory(tmp_path, name)


def test_creating_a_run_writes_its_source_and_settings(tmp_path):
    manager = props.PropsJobManager(tmp_path)
    job = manager.create("Grid Sheet.glb", b"glb", {"names": "barrel"})
    assert job.directory.name.startswith("Grid-Sheet__props__")
    assert job.source_glb.read_bytes() == b"glb"
    assert json.loads(job.settings_path.read_text())["names"] == ["barrel"]
    with pytest.raises(RuntimeError):
        manager.create("other.glb", b"glb", {})


def test_bad_settings_leave_no_directory_behind(tmp_path):
    with pytest.raises(ValueError):
        props.PropsJobManager(tmp_path).create("sheet.glb", b"glb", {"lods": "1,2"})
    assert not any(tmp_path.iterdir())


def test_turning_twice_adds_up_and_is_recorded(tmp_path):
    directory = _write_run(tmp_path)
    manager = props.PropsJobManager(tmp_path)
    first = manager.turn(directory.name, "chest", 90)
    assert first.only == "chest"
    assert first.settings["turns"] == {"chest": 90.0}
    first.status = "done"
    second = manager.turn(directory.name, "chest", 90)
    assert second.settings["turns"] == {"chest": 180.0}
    saved = json.loads((directory / "settings.json").read_text())
    assert saved["turns"] == {"chest": 180.0}


def test_four_quarter_turns_forget_the_turn(tmp_path):
    directory = _write_run(tmp_path)
    manager = props.PropsJobManager(tmp_path)
    for _ in range(4):
        job = manager.turn(directory.name, "chest", 90)
        job.status = "done"
    assert job.settings["turns"] == {}


@pytest.mark.parametrize(("prop", "degrees", "error"), [
    ("anvil", 90, RuntimeError), ("chest", 400, ValueError), ("chest", "lots", ValueError),
])
def test_a_turn_for_a_missing_prop_or_a_wild_angle_is_refused(tmp_path, prop, degrees, error):
    directory = _write_run(tmp_path)
    with pytest.raises(error):
        props.PropsJobManager(tmp_path).turn(directory.name, prop, degrees)


def test_generated_models_are_found_and_only_those_can_be_named(tmp_path):
    for relative in ("owl__pixal3d__1/owl__pixal3d__1.glb",
                     "research_only/grid__pixal3d__2/grid__pixal3d__2.glb",
                     "finish/x__finish__1/result.glb",
                     "props/sheet__props__1/finished/barrel/barrel_LOD0.glb"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"glb")
    found = sorted(m["path"] for m in props.generated_models(tmp_path))
    assert found == ["owl__pixal3d__1/owl__pixal3d__1.glb",
                     "research_only/grid__pixal3d__2/grid__pixal3d__2.glb"]
    assert props.generated_model(tmp_path, found[0]) == tmp_path / found[0]
    for stranger in ("../secret.glb", "finish/x__finish__1/result.glb", "/etc/passwd"):
        with pytest.raises(RuntimeError):
            props.generated_model(tmp_path, stranger)


def test_job_ids_are_validated_before_lookup():
    assert props.PropsJobManager().get("../../etc") is None
