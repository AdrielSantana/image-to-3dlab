"""Start, check and stop background jobs the same way on every platform.

Each viewer job runs in its own process group so stopping it also stops its children.
POSIX does that with sessions and `os.killpg`; Windows has neither, so it gets a new
process group at start and `taskkill /T` (kill the whole tree) at stop. Found on the first
real Windows run (issue #36): `os.killpg` does not exist there.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
from pathlib import PureWindowsPath

CREATE_NEW_PROCESS_GROUP = 0x00000200
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_IS_WINDOWS = os.name == "nt"
# Jobs the viewer starts: Python workers, and Blender for rig rebinds.
_OUR_IMAGES = ("python", "blender")


def group_popen_kwargs(windows: bool = _IS_WINDOWS) -> dict:
    """Popen keyword arguments that put the child in its own process group."""
    if windows:
        return {"creationflags": CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def windows_group_alive(pid: int, kernel32) -> bool:
    """True if `pid` is a running Python or Blender process. Windows recycles pids quickly, so a
    stale pid that now belongs to some other program must read as dead, or cleanup would
    kill that program's tree."""
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        if code.value != _STILL_ACTIVE:
            return False
        buffer = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_ulong(len(buffer))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return False
        return PureWindowsPath(buffer.value).name.lower().startswith(_OUR_IMAGES)
    finally:
        kernel32.CloseHandle(handle)


def group_alive(pid: int, windows: bool = _IS_WINDOWS) -> bool:
    """True if the process group led by `pid` still exists."""
    if windows:
        return windows_group_alive(pid, ctypes.windll.kernel32)
    try:
        os.killpg(pid, 0)  # signal 0: existence check only, sends nothing
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal


def terminate_group(pid: int, windows: bool = _IS_WINDOWS) -> None:
    """Best-effort stop of the group led by `pid`; one already gone is not an error."""
    if windows:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
