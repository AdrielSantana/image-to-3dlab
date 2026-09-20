"""Queued finishing jobs for the browser: retopologise, repaint, compress an existing GLB.

Deliberately a sibling of `rig_api.py` rather than part of `generate_api.py`: this operates
on an asset that already exists, so it has no image-to-3D backend, no model readiness and no
setup. What it shares is the job shape — one at a time, SSE progress, artifacts fetched by
URL — so the browser can drive it with the same code it already uses.

Finishing an asset the viewer generated earlier is the point: the surface response
(`metallic`/`roughness`/`ior`) is tuned per asset by eye, and re-finishing with different
values must not mean regenerating the geometry underneath it.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUTPUT_ROOT = REPO / "output" / "finish"
WORKER = REPO / "scripts" / "retopo_repaint.py"
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
TERMINAL = {"done", "error", "cancelled"}

ARTIFACTS = {
    "result.glb": ("result_glb", "model/gltf-binary"),
    "record.json": ("record_path", "application/json"),
}

# Bounds, not preferences. Everything here reaches a subprocess argument, so each value is
# clamped to a range the underlying scripts actually accept — `blender_retopo_bake.py`
# rejects a face target outside 1k..200k, and a voxel fraction outside its own range melts
# the subject or exhausts memory.
SETTING_BOUNDS: dict[str, tuple[float, float]] = {
    "faces": (1000, 200000),
    "atlas": (1024, 4096),
    "angle": (1.0, 89.9),
    "voxel": (0.0, 0.05),
    "metallic": (0.0, 1.0),
    "roughness": (0.0, 1.0),
    "ior": (1.0, 3.0),
    "paint_seed": (0, 2**31 - 1),
    "paint_res": (256, 1024),
    "paint_steps": (1, 50),
    "paint_tex": (1024, 4096),
    "texture_size": (256, 4096),
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "faces": 40000, "atlas": 2048, "angle": 89.0, "voxel": 0.004,
    "metallic": 0.25, "roughness": 0.65, "ior": 1.45,
    "paint_seed": 0, "paint_res": 512, "paint_steps": 15, "paint_tex": 4096,
    "texture_size": 2048, "skip_paint": False, "skip_compress": False,
}

INTEGER_SETTINGS = {
    "faces", "atlas", "paint_seed", "paint_res", "paint_steps", "paint_tex", "texture_size",
}

# What the progress bar does between stage markers. Retopology is a minute or two, the
# repaint is five or six, so the weights are deliberately uneven.
STAGE_PROGRESS = {"retopologise": 10, "repaint": 35, "compress": 90, "done": 100}


def normalise_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge client settings over the defaults, rejecting anything out of range.

    Raises ValueError with the offending key. Unknown keys are ignored rather than passed
    through — a typo must not silently reach a subprocess argument.
    """
    settings = dict(DEFAULT_SETTINGS)
    for key, value in raw.items():
        if key not in DEFAULT_SETTINGS:
            continue
        if key in {"skip_paint", "skip_compress"}:
            settings[key] = bool(value)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number, got {value!r}") from None
        low, high = SETTING_BOUNDS[key]
        if not low <= number <= high:
            raise ValueError(f"{key} must be within {low}..{high}, got {number}")
        settings[key] = int(number) if key in INTEGER_SETTINGS else number
    return settings


def build_command(job: FinishJob, settings: dict[str, Any]) -> list[str]:
    """The worker invocation. Flags, not positions — the worker owns its own ordering."""
    command = [
        sys.executable, "-u", str(WORKER),
        str(job.asset_path), str(job.image_path), str(job.result_glb),
        "--faces", str(settings["faces"]),
        "--atlas", str(settings["atlas"]),
        "--angle", str(settings["angle"]),
        "--voxel", str(settings["voxel"]),
        "--metallic", str(settings["metallic"]),
        "--roughness", str(settings["roughness"]),
        "--ior", str(settings["ior"]),
        "--paint-seed", str(settings["paint_seed"]),
        "--paint-res", str(settings["paint_res"]),
        "--paint-steps", str(settings["paint_steps"]),
        "--paint-tex", str(settings["paint_tex"]),
        "--texture-size", str(settings["texture_size"]),
    ]
    if settings["skip_paint"]:
        command.append("--skip-paint")
    if settings["skip_compress"]:
        command.append("--skip-compress")
    return command


class FinishJob:
    def __init__(self, job_id: str, directory: Path):
        self.id = job_id
        self.directory = directory
        self.asset_path = directory / "source.glb"
        self.image_path = directory / "source.png"
        self.result_glb = directory / "result.glb"
        self.record_path = directory / "result.retopo-repaint.json"
        self.settings: dict[str, Any] = dict(DEFAULT_SETTINGS)
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


class FinishJobManager:
    def __init__(self, output_root: Path = OUTPUT_ROOT):
        self.output_root = output_root
        self.jobs: dict[str, FinishJob] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def create(
        self, asset_name: str, asset: bytes, image: bytes, settings: dict[str, Any],
    ) -> FinishJob:
        with self.lock:
            active = self.jobs.get(self.active) if self.active else None
            if active is not None and active.status not in TERMINAL:
                raise RuntimeError("a finishing job is already running")
            stem = _slug(Path(asset_name).stem)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            directory = self.output_root / f"{stem}__finish__{stamp}"
            suffix = 2
            while directory.exists():
                directory = self.output_root / f"{stem}__finish__{stamp}-{suffix}"
                suffix += 1
            directory.mkdir(parents=True, exist_ok=False)
            job = FinishJob(uuid.uuid4().hex, directory)
            try:
                job.settings = normalise_settings(settings)
                job.asset_path.write_bytes(asset)
                job.image_path.write_bytes(image)
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def get(self, job_id: str) -> FinishJob | None:
        return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None

    def finish(self, job: FinishJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None


FINISH_JOBS = FinishJobManager()


def run_job(job: FinishJob, manager: FinishJobManager = FINISH_JOBS) -> None:
    if not WORKER.is_file():
        job.status = "error"
        job.emit({"phase": "error", "message": f"worker missing: {WORKER}"})
        manager.finish(job)
        return
    try:
        job.status = "running"
        job.emit({"phase": "queued", "overall_pct": 2, "message": "Starting"})
        job.process = subprocess.Popen(
            build_command(job, job.settings), cwd=str(REPO),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )
        job.directory.joinpath("pid").write_text(str(job.process.pid))
        assert job.process.stdout is not None
        for raw in job.process.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            job.append_log(line)
            event = stage_event(line)
            if event:
                job.emit(event)
        return_code = job.process.wait()
        job.append_log(f"worker exited with code {return_code}")
        if job.cancel_requested:
            job.status = "cancelled"
            job.emit({"phase": "error", "message": "Finishing cancelled"})
        elif return_code != 0:
            job.status = "error"
            job.emit({
                "phase": "error", "message": f"worker exited with code {return_code}",
                "log_tail": "\n".join(job.log_lines)[-8000:],
            })
        elif job.result_glb.is_file():
            job.status = "done"
            job.emit({
                "phase": "done", "overall_pct": 100,
                "message": f"Finished at {job.result_glb.stat().st_size / 1048576:.1f} MB",
                "result_url": f"/api/finish/{job.id}/result.glb",
                "record_url": f"/api/finish/{job.id}/record.json",
                "size_bytes": job.result_glb.stat().st_size,
            })
        else:
            job.status = "error"
            job.emit({"phase": "error", "message": "worker exited without writing a GLB"})
    except Exception as exc:
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc)})
    finally:
        job.directory.joinpath("pid").unlink(missing_ok=True)
        manager.finish(job)


def cancel_job(job: FinishJob) -> None:
    if job.status in TERMINAL:
        raise RuntimeError(f"job is already {job.status}")
    job.cancel_requested = True
    job.status = "cancelling"
    if job.process is not None and job.process.poll() is None:
        try:
            os.killpg(job.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def status_payload(job: FinishJob) -> dict[str, Any]:
    return {
        "status": job.status,
        "settings": job.settings,
        "last_event": job.events[-1] if job.events else None,
    }


def stage_event(line: str) -> dict[str, Any] | None:
    """Turn one `I2L_STAGE::phase::message` line into an SSE event, or None."""
    if not line.startswith("I2L_STAGE::"):
        return None
    _, phase, message = line.split("::", 2)
    return {"phase": phase, "overall_pct": STAGE_PROGRESS.get(phase, 5), "message": message}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:80] or "asset"
