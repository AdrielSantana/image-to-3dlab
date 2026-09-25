"""Queued prop-sheet jobs for the browser: split one GLB into props, then give each LODs.

A sibling of `finish_api.py`, with the same job shape (one at a time, SSE progress, runs
kept on disk), driving the two scripts `docs/prop-sheets.md` walks through:
`scripts/blender_split_props.py` takes the sheet apart, and `scripts/finish_props.py`
bakes each prop's LODs and compresses them. Nothing here touches the GPU: the split is
headless Blender, and the LOD bakes are Blender's CPU bake.

A run is re-openable from its directory alone, so the result list is built from what is
on disk rather than from this process's memory. That is also how a prop that came back
facing sideways is fixed: "turn" re-splits the sheet with the extra turn and re-bakes
that one prop, in the same directory, in about ten seconds plus its LODs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from itertools import pairwise
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finish_api import remaining_seconds, served_url  # noqa: E402
from rig_api import blender_executable  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from image_to_3dlab import processes  # noqa: E402

OUTPUT_ROOT = REPO / "output" / "props"
SPLITTER = REPO / "scripts" / "blender_split_props.py"
FINISHER = REPO / "scripts" / "finish_props.py"
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
# The exact shape `create` builds. A turn takes its directory name from a URL, so the
# name is matched against the generator rather than merely sanitised.
RUN_DIRECTORY = re.compile(r"^[A-Za-z0-9_-]{1,80}__props__\d{8}-\d{6}(?:-\d+)?$")
# A prop name becomes a file name and a command-line argument.
PROP_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
TERMINAL = {"done", "error", "cancelled"}
MAX_PROPS = 64
MAX_LODS = 4

# `scripts/blender_split_props.py`'s YAW_THRESHOLD, repeated because that script imports
# numpy at module level and this server must not. A test keeps the two equal.
YAW_THRESHOLD = 0.10

DEFAULT_SETTINGS: dict[str, Any] = {
    "names": [], "lods": [5000, 2500, 1000], "atlas": 1024,
    "metallic": 0.25, "roughness": 0.65, "ior": 1.45,
    "yaw_threshold": YAW_THRESHOLD, "compress": True, "turns": {},
}
SETTING_BOUNDS: dict[str, tuple[float, float]] = {
    "metallic": (0.0, 1.0), "roughness": (0.0, 1.0), "ior": (1.0, 3.0),
    "yaw_threshold": (0.0, 0.99),
}
ATLAS_SIZES = (1024, 2048, 4096)
FACE_RANGE = (1000, 200000)

# How the bar divides. A measured nine-prop run on 2026-09-25 split in about 8 seconds
# and baked 27 LODs in 133, so the split gets a thin slice and the props share the rest.
SPLIT_WEIGHT = 6.0

# Top-level directories under output/ that hold something other than generated models.
NOT_GENERATED = {"finish", "props", "images", "rig-rebind"}

# What `finish_props.py` prints as it starts a bake: `[barrel LOD1] /path/to/blender ...`.
# Its gltfpack step prints `[barrel LOD1 gltfpack] ...`, which this deliberately does
# not match: the bake is the slow part, and counting both would make the bar lurch.
BAKE_START = re.compile(r"^\[(?P<name>[A-Za-z0-9_-]+) LOD(?P<index>\d+)\] ")
# And when a prop is done: `barrel: LOD0 425 KB, LOD1 387 KB, LOD2 327 KB`.
PROP_DONE = re.compile(r"^(?P<name>[A-Za-z0-9_-]+): LOD0 ")


def normalise_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge client settings over the defaults, rejecting anything the scripts would.

    Raises ValueError naming the offending key. Unknown keys are dropped, not forwarded:
    a typo must not reach a command line as a stray flag.
    """
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    for key, value in raw.items():
        if key not in DEFAULT_SETTINGS:
            continue
        if key == "names":
            settings[key] = normalise_names(value)
        elif key == "lods":
            settings[key] = normalise_lods(value)
        elif key == "turns":
            settings[key] = normalise_turns(value)
        elif key == "compress":
            settings[key] = bool(value)
        elif key == "atlas":
            atlas = _number(key, value)
            if atlas not in ATLAS_SIZES:
                raise ValueError(f"atlas must be one of {ATLAS_SIZES}, got {value!r}")
            settings[key] = int(atlas)
        else:
            number = _number(key, value)
            low, high = SETTING_BOUNDS[key]
            if not low <= number <= high:
                raise ValueError(f"{key} must be within {low}..{high}, got {number}")
            settings[key] = number
    return settings


def normalise_names(value: Any) -> list[str]:
    """Names in reading order. A string is one name per line; blank lines are skipped."""
    if isinstance(value, str):
        value = value.splitlines()
    if not isinstance(value, list):
        raise ValueError("names must be a list or one name per line")
    names = [str(name).strip() for name in value if str(name).strip()]
    if len(names) > MAX_PROPS:
        raise ValueError(f"names: at most {MAX_PROPS}, got {len(names)}")
    for name in names:
        if not PROP_NAME.fullmatch(name):
            raise ValueError(f"names: {name!r} must be letters, digits, - or _ (up to 40)")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"names: each name once, {', '.join(duplicates)} repeated")
    return names


def normalise_lods(value: Any) -> list[int]:
    """Face counts, most detailed first, the same rule `finish_props.parse_lods` applies."""
    if isinstance(value, str):
        value = [part for part in value.split(",") if part.strip()]
    if not isinstance(value, list) or not value:
        raise ValueError("lods must be a list of face counts")
    if len(value) > MAX_LODS:
        raise ValueError(f"lods: at most {MAX_LODS}, got {len(value)}")
    lods = [int(_number("lods", faces)) for faces in value]
    low, high = FACE_RANGE
    for faces in lods:
        if not low <= faces <= high:
            raise ValueError(f"lods: each must be {low}..{high} faces, got {faces}")
    if any(later >= earlier for earlier, later in pairwise(lods)):
        raise ValueError(f"lods go from most detailed to least, got {lods}")
    return lods


def normalise_turns(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("turns must map a prop name to degrees")
    turns = {}
    for name, degrees in value.items():
        if not PROP_NAME.fullmatch(str(name)):
            raise ValueError(f"turns: {name!r} is not a prop name")
        turns[str(name)] = wrap_degrees(_number("turns", degrees))
    return turns


def wrap_degrees(degrees: float) -> float:
    """Into (-180, 180], so four quarter turns come back to none rather than 360."""
    wrapped = (degrees + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _number(key: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {value!r}") from None


def finisher_module():
    """`finish_props.py`, imported, for its gltfpack lookup and file layout.

    Its module level is constants and small functions; importing it runs nothing.
    """
    spec = importlib.util.spec_from_file_location("finish_props", FINISHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_gltfpack() -> Path | None:
    return finisher_module().find_gltfpack()


def split_command(job: PropsJob, blender: Path) -> list[str]:
    command = [
        str(blender), "-b", "--factory-startup", "-P", str(SPLITTER), "--",
        str(job.source_glb), str(job.split_dir),
        "--yaw-threshold", str(job.settings["yaw_threshold"]),
    ]
    if job.settings["names"]:
        command += ["--names", *job.settings["names"]]
    for name, degrees in job.settings["turns"].items():
        command += ["--turn", f"{name}={degrees:g}"]
    return command


def finish_command(
    job: PropsJob, blender: Path, gltfpack: Path | None, only: str | None = None,
) -> list[str]:
    """Bake every prop the split wrote, or just `only` when a turn re-does one."""
    source = job.split_dir / f"{only}.glb" if only else job.split_dir
    settings = job.settings
    command = [
        sys.executable, "-u", str(FINISHER), str(source), str(job.finished_dir),
        "--lods", ",".join(map(str, settings["lods"])),
        "--atlas", str(settings["atlas"]),
        "--metallic", str(settings["metallic"]),
        "--roughness", str(settings["roughness"]),
        "--ior", str(settings["ior"]),
        "--blender", str(blender),
    ]
    if settings["compress"] and gltfpack is not None:
        command += ["--gltfpack", str(gltfpack)]
    else:
        command.append("--no-compress")
    return command


def stage_meta(names: list[str]) -> dict[str, Any]:
    """The panel's rows: the split, then one row per prop. Prefixed so a prop called
    `split` or `done` cannot collide with a phase the panel treats specially."""
    stages = ["split", *(f"prop:{name}" for name in names)]
    labels = {"split": "Split the sheet", **{f"prop:{name}": name for name in names}}
    return {"stages": stages, "stage_labels": labels}


class PropsProgress:
    """Turn the two scripts' output into progress events.

    The split has no sub-progress worth showing (it takes seconds). Once it has written
    its props, the rest of the bar is shared equally between them, and each prop's row
    counts its LODs: `finish_props.py` announces every bake as it starts, so a prop on
    its second of three LODs has one done.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.names: list[str] = []
        self.lods = 1
        self.done: dict[str, int] = {}
        self.stage_started = self.started

    def begin_split(self, message: str = "Splitting the sheet") -> dict[str, Any]:
        return self._event("split", message, 0.0)

    def split_line(self, line: str) -> dict[str, Any] | None:
        if not line.startswith("SPLIT::"):
            return None
        return self._event("split", line[len("SPLIT::"):].strip(), 0.0)

    def begin_props(self, names: list[str], lods: int) -> dict[str, Any]:
        self.names = list(names)
        self.lods = max(lods, 1)
        self.done = {name: 0 for name in self.names}
        first = self.names[0] if self.names else "split"
        event = self._event(f"prop:{first}" if self.names else "split",
                            f"{len(self.names)} props to bake", SPLIT_WEIGHT / 100.0)
        event.update(stage_meta(self.names))
        return event

    def finish_line(self, line: str) -> dict[str, Any] | None:
        start = BAKE_START.match(line)
        if start and start.group("name") in self.done:
            name, index = start.group("name"), int(start.group("index"))
            self.done[name] = max(self.done[name], min(index, self.lods))
            if index == 0:
                self.stage_started = self.clock()
            return self._prop_event(name, f"{name}: baking LOD{index}")
        finished = PROP_DONE.match(line)
        if finished and finished.group("name") in self.done:
            name = finished.group("name")
            self.done[name] = self.lods
            return self._prop_event(name, line.strip())
        return None

    def fraction(self) -> float:
        if not self.names:
            return SPLIT_WEIGHT / 100.0
        baked = sum(self.done.values()) / (len(self.names) * self.lods)
        return (SPLIT_WEIGHT + (100.0 - SPLIT_WEIGHT) * baked) / 100.0

    def _prop_event(self, name: str, message: str) -> dict[str, Any]:
        step = self.done[name]
        event = self._event(f"prop:{name}", message, self.fraction(),
                            step=step, total=self.lods)
        event["stage_pct"] = round(100 * step / self.lods)
        event["stage_eta_seconds"] = remaining_seconds(
            self.clock() - self.stage_started, step / self.lods,
        )
        return event

    def _event(self, phase: str, message: str, fraction: float, **extra: Any) -> dict[str, Any]:
        return {
            "phase": phase,
            "message": message,
            "overall_pct": round(100.0 * fraction, 1),
            "total_eta_seconds": remaining_seconds(self.clock() - self.started, fraction),
            **extra,
        }


class PropsJob:
    def __init__(self, job_id: str, directory: Path):
        self.id = job_id
        self.directory = directory
        self.source_glb = directory / "source.glb"
        self.settings_path = directory / "settings.json"
        self.split_dir = directory / "split"
        self.finished_dir = directory / "finished"
        self.settings: dict[str, Any] = json.loads(json.dumps(DEFAULT_SETTINGS))
        self.only: str | None = None   # set for a turn: re-bake just this prop
        self.status = "queued"
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.log_lines: deque[str] = deque(maxlen=300)

    def emit(self, event: dict[str, Any]) -> None:
        payload = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(payload)
            self.condition.notify_all()

    def append_log(self, line: str) -> None:
        self.log_lines.append(line)
        with self.directory.joinpath("run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class PropsJobManager:
    def __init__(self, output_root: Path = OUTPUT_ROOT):
        self.output_root = output_root
        self.jobs: dict[str, PropsJob] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def _refuse_if_busy(self) -> None:
        active = self.jobs.get(self.active) if self.active else None
        if active is not None and active.status not in TERMINAL:
            raise RuntimeError("a prop-sheet job is already running")

    def create(self, asset_name: str, asset: bytes, settings: dict[str, Any]) -> PropsJob:
        with self.lock:
            self._refuse_if_busy()
            clean = normalise_settings(settings)
            stem = _slug(Path(asset_name).stem)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            directory = self.output_root / f"{stem}__props__{stamp}"
            suffix = 2
            while directory.exists():
                directory = self.output_root / f"{stem}__props__{stamp}-{suffix}"
                suffix += 1
            directory.mkdir(parents=True, exist_ok=False)
            job = PropsJob(uuid.uuid4().hex, directory)
            try:
                job.settings = clean
                job.source_glb.write_bytes(asset)
                job.settings_path.write_text(json.dumps(job.settings, indent=2))
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def turn(self, name: str, prop: str, degrees: float) -> PropsJob:
        """A job that turns one prop of an existing run and re-bakes only that prop.

        The turn is added to the run's recorded turns, so turning the chest twice by 90
        faces it backwards, as clicking the button twice should.
        """
        with self.lock:
            self._refuse_if_busy()
            directory = run_directory(self.output_root, name)
            job = PropsJob(uuid.uuid4().hex, directory)
            if not job.source_glb.is_file() or not job.settings_path.is_file():
                raise RuntimeError(f"{name} is missing its source.glb or settings.json")
            settings = normalise_settings(json.loads(job.settings_path.read_text()))
            if prop not in prop_names(directory):
                raise RuntimeError(f"{name} has no prop called {prop!r}")
            step = _number("degrees", degrees)
            if not -180.0 <= step <= 180.0:
                raise ValueError(f"degrees must be within -180..180, got {step}")
            turns = dict(settings["turns"])
            turns[prop] = wrap_degrees(turns.get(prop, 0.0) + step)
            if turns[prop] == 0.0:
                del turns[prop]
            settings["turns"] = turns
            job.settings = settings
            job.only = prop
            job.settings_path.write_text(json.dumps(settings, indent=2))
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def get(self, job_id: str) -> PropsJob | None:
        return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None

    def finish(self, job: PropsJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None


PROPS_JOBS = PropsJobManager()


def run_job(job: PropsJob, manager: PropsJobManager = PROPS_JOBS) -> None:
    try:
        blender = blender_executable()
        if blender is None:
            raise RuntimeError("Blender not found: install it or set I2L_BLENDER")
        for script in (SPLITTER, FINISHER):
            if not script.is_file():
                raise RuntimeError(f"script missing: {script}")
        job.status = "running"
        progress = PropsProgress()
        rows = [job.only] if job.only else []
        job.emit({**progress.begin_split(), **stage_meta(rows)})

        if not _run_step(job, "split", split_command(job, blender), progress.split_line):
            return
        names = prop_names(job.directory)
        if not names:
            raise RuntimeError("the split found no props in this GLB")
        if job.only:
            names = [job.only]
            shutil.rmtree(job.finished_dir / job.only, ignore_errors=True)

        gltfpack = find_gltfpack() if job.settings["compress"] else None
        event = progress.begin_props(names, len(job.settings["lods"]))
        if job.settings["compress"] and gltfpack is None:
            event["message"] += "; gltfpack not found, so no .web.glb files"
        job.emit(event)
        if not _run_step(job, "LODs", finish_command(job, blender, gltfpack, job.only),
                         progress.finish_line):
            return

        run = describe_run(job.directory)
        job.status = "done"
        job.emit({
            "phase": "done", "overall_pct": 100,
            "message": (f"Turned and re-baked {job.only}" if job.only
                        else f"{len(run['props'])} props finished"),
            "directory": job.directory.name,
            "run": run,
        })
    except Exception as exc:
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc),
                  "log_tail": "\n".join(job.log_lines)[-8000:]})
    finally:
        job.directory.joinpath("pid").unlink(missing_ok=True)
        manager.finish(job)


def _run_step(job: PropsJob, label: str, command: list[str], feed) -> bool:
    """One subprocess, its output logged and fed to the progress parser.

    False, with the job's terminal event already emitted, when it did not succeed.
    """
    job.process = subprocess.Popen(
        command, cwd=str(REPO), env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        **processes.group_popen_kwargs(),
    )
    job.directory.joinpath("pid").write_text(str(job.process.pid))
    assert job.process.stdout is not None
    for raw in job.process.stdout:
        line = raw.rstrip("\n")
        if not line:
            continue
        job.append_log(line)
        event = feed(line)
        if event:
            job.emit(event)
    code = job.process.wait()
    job.append_log(f"{label} exited with code {code}")
    if job.cancel_requested:
        job.status = "cancelled"
        job.emit({"phase": "cancelled", "message": "Cancelled"})
        return False
    if code != 0:
        job.status = "error"
        job.emit({"phase": "error", "message": f"{label} exited with code {code}",
                  "log_tail": "\n".join(job.log_lines)[-8000:]})
        return False
    return True


def cancel_job(job: PropsJob) -> None:
    if job.status in TERMINAL:
        raise RuntimeError(f"job is already {job.status}")
    job.cancel_requested = True
    job.status = "cancelling"
    if job.process is not None and job.process.poll() is None:
        processes.terminate_group(job.process.pid)


def status_payload(job: PropsJob) -> dict[str, Any]:
    return {
        "status": job.status,
        "settings": job.settings,
        "directory": job.directory.name,
        "log_tail": "\n".join(job.log_lines)[-4000:],
        "last_event": job.events[-1] if job.events else None,
    }


def run_directory(root: Path, name: str) -> Path:
    """One run directory under `root`, by name, with no way out of it."""
    if not RUN_DIRECTORY.fullmatch(name):
        raise RuntimeError(f"not a run directory name: {name!r}")
    directory = root / name
    if not directory.is_dir():
        raise RuntimeError(f"no such run: {name}")
    return directory


def prop_names(directory: Path) -> list[str]:
    """The props the split wrote, in reading order, from its own record."""
    record = _read_json(directory / "split" / "props.json") or {}
    return [entry["name"] for entry in record.get("props", [])
            if isinstance(entry, dict) and PROP_NAME.fullmatch(str(entry.get("name", "")))]


def describe_run(directory: Path) -> dict[str, Any]:
    """Everything the page shows about one run, read from disk.

    The LODs come from scanning `finished/`, not from `finish_props.json`: a turn
    re-bakes one prop and rewrites that record with only that prop in it.
    """
    split = _read_json(directory / "split" / "props.json") or {}
    settings = _read_json(directory / "settings.json") or {}
    props = []
    for entry in split.get("props", []):
        name = str(entry.get("name", ""))
        if not PROP_NAME.fullmatch(name):
            continue
        lods = []
        for index in range(MAX_LODS):
            plain = directory / "finished" / name / f"{name}_LOD{index}.glb"
            if not plain.is_file():
                break
            web = plain.with_name(f"{name}_LOD{index}.web.glb")
            lods.append({
                "index": index,
                "url": served_url(plain),
                "bytes": plain.stat().st_size,
                "web_url": served_url(web) if web.is_file() else None,
                "web_bytes": web.stat().st_size if web.is_file() else None,
            })
        # The plain LOD0, not the .web one: the viewer has no meshopt decoder, and the two
        # look the same. Before its LODs exist, the split prop itself.
        split_glb = directory / "split" / f"{name}.glb"
        preview = lods[0]["url"] if lods else served_url(split_glb) if split_glb.is_file() else None
        props.append({
            "name": name,
            "faces": entry.get("faces"),
            "size": entry.get("size"),
            "tilt_degrees": entry.get("tilt_degrees"),
            "yaw_degrees": entry.get("yaw_degrees", 0.0),
            "yaw_tie": bool(entry.get("yaw_tie")),
            "extra_turn_degrees": entry.get("extra_turn_degrees", 0.0),
            "lods": lods,
            "preview_url": preview,
        })
    finished = bool(props) and all(len(p["lods"]) == len(settings.get("lods", [1])) for p in props)
    return {
        "directory": directory.name,
        "modified": directory.stat().st_mtime,
        "settings": settings,
        "props": props,
        "finished": finished,
        "blend_url": served_url(directory / "split" / "props.blend")
        if (directory / "split" / "props.blend").is_file() else None,
    }


def list_runs(root: Path = OUTPUT_ROOT, limit: int = 15) -> list[dict[str, Any]]:
    """Every prop-sheet run on disk, newest first."""
    if not root.is_dir():
        return []
    directories = sorted(
        (d for d in root.iterdir() if d.is_dir() and RUN_DIRECTORY.fullmatch(d.name)),
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    return [describe_run(d) for d in directories[:limit]]


def generated_models(root: Path, limit: int = 25) -> list[dict[str, Any]]:
    """GLBs the Generate tab made, newest first: `<name>/<name>.glb` one or two levels
    under output/ (a licence-class folder may sit in between)."""
    if not root.is_dir():
        return []
    found = []
    for top in root.iterdir():
        if not top.is_dir() or top.name in NOT_GENERATED or top.name.startswith("."):
            continue
        for directory in (top, *(d for d in top.iterdir() if d.is_dir())):
            glb = directory / f"{directory.name}.glb"
            if glb.is_file():
                found.append(glb)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.stem, "path": p.relative_to(root).as_posix(),
             "bytes": p.stat().st_size, "url": served_url(p)} for p in found[:limit]]


def generated_model(root: Path, relative: str) -> Path:
    """One of `generated_models`, by the path the page was given, and nothing else."""
    for model in generated_models(root, limit=10_000):
        if model["path"] == relative:
            return root / relative
    raise RuntimeError(f"not a generated model: {relative!r}")


def tools_payload() -> dict[str, Any]:
    """What the page must know before offering a run: can it split, and can it compress."""
    blender = blender_executable()
    try:
        gltfpack = find_gltfpack()
    except SystemExit:
        gltfpack = None
    return {"blender": str(blender) if blender else None,
            "gltfpack": str(gltfpack) if gltfpack else None}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:80] or "sheet"
