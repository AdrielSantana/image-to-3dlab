"""Run one backend's weight download, and report honestly on how it is going.

`AGENTS.md` forbids fetching weights before the user has confirmed which pipeline and
which route. The Setup & Status page is where they confirm; this is what runs afterwards.

**Progress is measured, not parsed.** `huggingface_hub` draws tqdm bars with carriage
returns and ANSI escapes, which pipe into a browser as garbage, and each backend's
bootstrap prints something different anyway. Instead the target directory is polled and
compared against the size the catalogue already knows, which gives one progress mechanism
for all three backends, survives a resumed download, and does not care what the tool
prints. The log is kept alongside, stripped of escape codes, for diagnosis.

**Three signals, because one is not enough.** Directory growth says whether bytes are
arriving; the process exit code says whether it worked; the log tail says why it did not.
A download that has grown by nothing for `STALL_SECONDS` is reported as stalled rather
than left to look like slow progress, which is the failure people actually hit.
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
from collections import deque
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))

from backend_catalog import (  # noqa: E402
    BY_ID,
    HF_HUB_DIR,
    Backend,
    human_bytes,
    runs_on_phrase,
    venv_python,
)

POLL_SECONDS = 2.0
STALL_SECONDS = 90.0
ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\r")
TERMINAL = {"done", "error", "cancelled"}

# How each backend is installed. Kept here rather than in the catalogue because the
# catalogue describes *what* a backend needs and this describes *how* to get it; the
# viewer shows the first to everyone and only ever runs the second on request.
COMMANDS: dict[str, list[str]] = {
    "trellis": [sys.executable, str(REPO / "scripts" / "bootstrap_trellis_space_macos.py")],
    "pixal3d": [sys.executable, str(REPO / "scripts" / "bootstrap_pixal3d.py"), "--yes"],
    "hunyuan_xiong": [
        str(venv_python(REPO / "hunyuan_mlx" / "shape")),
        str(REPO / "hunyuan_mlx" / "download_weights.py"),
        # Explicitly the default route, not every model. Without --model this fetches all
        # three shape checkpoints, which is 23 GB where the default route needs 5.
        "--model", "2.0",
    ],
    # --yes because the browser already asked. The confirmation AGENTS.md requires is the
    # Setup & Status dialog; asking again on a stdin nobody is attached to would hang.
    "qwen-image": [sys.executable, str(REPO / "scripts" / "bootstrap_qwen_image.py"),
                   "--yes"],
}


def strip_ansi(line: str) -> str:
    """Terminal output made readable in a browser.

    Progress bars redraw with carriage returns, so the last segment of a `\\r`-split line
    is the only current one; everything before it is a frame nobody needs to see again.
    """
    return ANSI.sub("\n", line).strip().split("\n")[-1].strip()


def rate_and_eta(
    samples: list[tuple[float, int]], remaining: int,
) -> tuple[float | None, float | None]:
    """Bytes per second over the sample window, and seconds left at that rate.

    Measured across the window rather than since the start, so pausing early does not
    depress the estimate for the rest of the run.
    """
    if len(samples) < 2:
        return None, None
    (t0, b0), (t1, b1) = samples[0], samples[-1]
    seconds = t1 - t0
    grown = b1 - b0
    if seconds <= 0 or grown <= 0:
        return 0.0, None
    rate = grown / seconds
    return rate, (remaining / rate if remaining > 0 else 0.0)


def describe_progress(
    backend: Backend, present: int, rate: float | None, eta: float | None, stalled: bool,
) -> dict[str, Any]:
    expected = backend.bytes_expected
    percent = 0 if expected <= 0 else max(0, min(99, round(present / expected * 100)))
    if stalled:
        detail = f"stalled — no new data for {int(STALL_SECONDS)}s ({human_bytes(present)} so far)"
    elif rate:
        eta_text = f" · ~{int(eta // 60)} min left" if eta and eta > 60 else (
            f" · ~{int(eta)}s left" if eta else "")
        detail = (f"{human_bytes(present)} of {human_bytes(expected)}"
                  f" · {human_bytes(int(rate))}/s{eta_text}")
    else:
        detail = f"{human_bytes(present)} of {human_bytes(expected)}"
    return {"phase": "downloading", "overall_pct": percent, "detail": detail,
            "bytes_present": present, "stalled": stalled}


class DownloadRun:
    """One bootstrap subprocess, with the SSE surface the other job types use."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.status = "queued"
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.process: subprocess.Popen[bytes] | None = None
        self.cancelled = False
        self.log: deque[str] = deque(maxlen=400)

    def emit(self, event: dict[str, Any]) -> None:
        payload = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(payload)
            self.condition.notify_all()

    def bytes_present(self) -> int:
        from backend_catalog import _dir_state
        return sum(_dir_state(w.path)[1] for w in self.backend.weights)


DOWNLOADS: dict[str, DownloadRun] = {}
LOCK = threading.Lock()


def active() -> DownloadRun | None:
    return next((r for r in DOWNLOADS.values() if r.status not in TERMINAL), None)


def start(backend_id: str) -> DownloadRun:
    backend = BY_ID.get(backend_id)
    if backend is None:
        raise KeyError(f"unknown backend: {backend_id}")
    if backend_id not in COMMANDS:
        raise RuntimeError(f"{backend.label} has no automated setup yet")
    # Checked here rather than only in the browser, because the API is the thing that
    # spends someone's bandwidth and an unsupported machine cannot finish the job.
    if not backend.runs_here():
        raise RuntimeError(
            f"{backend.label} needs {runs_on_phrase(backend)}, which this machine is not. "
            f"Nothing has been downloaded."
        )
    with LOCK:
        if active() is not None:
            raise RuntimeError("a download is already running")
        run = DownloadRun(backend)
        DOWNLOADS[backend_id] = run
    threading.Thread(target=_run, args=(run,), daemon=True,
                     name=f"download-{backend_id}").start()
    return run


def cancel(backend_id: str) -> None:
    run = DOWNLOADS.get(backend_id)
    if run is None or run.status in TERMINAL:
        raise RuntimeError("no download is running for that backend")
    run.cancelled = True
    if run.process is not None and run.process.poll() is None:
        try:
            if os.name == "nt":
                run.process.terminate()
            else:
                os.killpg(run.process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def remove(backend_id: str) -> dict[str, Any]:
    """Delete one backend's weights from disk.

    Deliberately separate from cancelling. A cancelled download leaves partial files that
    Hugging Face resumes from, so throwing them away automatically would turn a pause into
    a restart; someone who wants the space back, or who has a corrupt download to clear,
    asks for that explicitly.

    Only paths the catalogue declares are touched, and only inside the repository or the
    Hugging Face cache, so a bad entry cannot make this delete something else.
    """
    backend = BY_ID.get(backend_id)
    if backend is None:
        raise KeyError(f"unknown backend: {backend_id}")
    run = DOWNLOADS.get(backend_id)
    if run is not None and run.status not in TERMINAL:
        raise RuntimeError("that backend is downloading right now; cancel it first")

    freed, removed = 0, []
    for weight in backend.weights:
        path = weight.path.resolve()
        if not _inside_known_roots(path):
            raise RuntimeError(f"refusing to delete outside the repo or cache: {path}")
        if not path.is_dir():
            continue
        from backend_catalog import _dir_state
        freed += _dir_state(path)[1]
        shutil.rmtree(path)
        removed.append(str(path))
    return {"backend": backend_id, "freed_bytes": freed,
            "freed": human_bytes(freed), "removed": removed}


def _inside_known_roots(path: Path) -> bool:
    roots = (REPO.resolve(), HF_HUB_DIR.resolve())
    return any(root == path or root in path.parents for root in roots)


def status_payload(run: DownloadRun) -> dict[str, Any]:
    return {
        "backend": run.backend.id,
        "status": run.status,
        "last_event": run.events[-1] if run.events else None,
        "log_tail": "\n".join(run.log)[-4000:],
    }


def _missing_executable(program: str) -> str | None:
    """Why a command cannot start, said in words, before the OS says it in error codes.

    `Popen` on a path that is not there raises `FileNotFoundError`, whose message reaches
    the browser as "[Errno 2] No such file or directory" -- or, reported for real from a
    Windows machine, "[WinError 2] The system cannot find the file specified". Neither says
    which file or what to do about it. Checked first, it can.
    """
    if shutil.which(program) or Path(program).exists():
        return None
    path = Path(program)
    venv = next((parent for parent in path.parents if parent.name == ".venv"), None)
    if venv is not None:
        return (f"the Python environment at {venv} has not been created yet. "
                f"Run `uv sync` in {venv.parent}, then try again.")
    return f"`{program}` is not installed, or not on this machine's PATH."


def _run(run: DownloadRun) -> None:
    run.status = "running"
    missing = _missing_executable(COMMANDS[run.backend.id][0])
    if missing is not None:
        run.status = "error"
        run.emit({"phase": "error", "overall_pct": 0, "detail": f"cannot start: {missing}"})
        return
    run.emit({"phase": "queued", "overall_pct": 0,
              "detail": f"starting · {human_bytes(run.backend.bytes_expected)} expected"})
    stop = threading.Event()
    if run.backend.setup_fetches_weights:
        threading.Thread(target=_watch_size, args=(run, stop), daemon=True).start()
    else:
        threading.Thread(target=_watch_elapsed, args=(run, stop), daemon=True).start()
    try:
        run.process = subprocess.Popen(
            COMMANDS[run.backend.id], cwd=str(REPO),
            env={**os.environ, "PYTHONUNBUFFERED": "1",
                 # The bars are unreadable in a browser and the size watcher is the real
                 # progress signal, so ask the downloader not to draw them at all.
                 "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
            # Its own process group, so cancelling kills the downloader's children too.
            # POSIX-only: Windows rejects the argument outright, and `cancel` falls back to
            # terminating the process there.
            **({"start_new_session": True} if os.name != "nt" else {}),
        )
        assert run.process.stdout is not None
        for raw in run.process.stdout:
            line = strip_ansi(raw.decode("utf-8", errors="replace"))
            if line:
                run.log.append(line)
                run.emit({"phase": "log", "log": line})
        code = run.process.wait()
    except Exception as exc:  # noqa: BLE001 - reported to the browser, not swallowed
        stop.set()
        run.status = "error"
        run.emit({"phase": "error", "detail": str(exc)})
        return
    finally:
        stop.set()

    present = run.bytes_present()
    if run.cancelled:
        run.status = "cancelled"
        run.emit({"phase": "cancelled",
                  "detail": f"cancelled · {human_bytes(present)} kept, "
                            f"resuming will continue from here"})
    elif code != 0:
        run.status = "error"
        run.emit({"phase": "error", "overall_pct": 0,
                  "detail": _explain(code, list(run.log))})
    else:
        run.status = "done"
        fetched = run.backend.setup_fetches_weights
        run.emit({"phase": "done", "overall_pct": 100,
                  "detail": f"done · {human_bytes(present)} on disk" if fetched else
                            "built · weights download on the first generation run"})


def _watch_size(run: DownloadRun, stop: threading.Event) -> None:
    """Poll the target directories and report growth, rate and stalls."""
    samples: list[tuple[float, int]] = []
    last_growth = time.monotonic()
    last_bytes = run.bytes_present()
    while not stop.wait(POLL_SECONDS):
        now = time.monotonic()
        present = run.bytes_present()
        samples.append((now, present))
        del samples[:-15]
        if present > last_bytes:
            last_growth = now
            last_bytes = present
        remaining = max(0, run.backend.bytes_expected - present)
        rate, eta = rate_and_eta(samples, remaining)
        run.emit(describe_progress(
            run.backend, present, rate, eta, stalled=now - last_growth > STALL_SECONDS,
        ))


def _watch_elapsed(run: DownloadRun, stop: threading.Event) -> None:
    """Progress for a step that builds rather than downloads.

    There is nothing to measure -- no directory grows -- so this reports elapsed time
    against the catalogue's estimate and never claims a stall. Saying "stalled" during a
    healthy hour-long compile is worse than saying nothing.
    """
    estimate = (run.backend.setup_minutes or 0) * 60
    while not stop.wait(POLL_SECONDS * 2):
        elapsed = time.monotonic() - run.started
        percent = 0 if estimate <= 0 else max(0, min(95, round(elapsed / estimate * 100)))
        run.emit({
            "phase": "building", "overall_pct": percent,
            "detail": f"building the Metal port · {int(elapsed // 60)} min elapsed"
                      + (f" of roughly {estimate // 60:.0f}" if estimate else ""),
        })


def _explain(code: int, log: list[str]) -> str:
    """Turn an exit code into something a person can act on.

    A bare "exited with code 1" sends someone to the log to work out whether they are
    offline, unauthorised or out of disk. These three cover what actually happens.
    """
    tail = "\n".join(log[-12:]).lower()
    if "401" in tail or "gated" in tail or "authenticate" in tail or "token" in tail:
        return ("refused: this model needs a Hugging Face login and its terms accepted. "
                "Run `huggingface-cli login`, accept the terms on the model page, and retry.")
    if "no space left" in tail or "enospc" in tail:
        return "ran out of disk space. Free some room and retry; what downloaded is kept."
    if "temporary failure in name resolution" in tail or "connection" in tail:
        return "network error. Check the connection and retry; what downloaded is kept."
    last = next((line for line in reversed(log) if line.strip()), "")
    return f"setup exited with code {code}. Last line: {last}"
