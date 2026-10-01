"""scripts/bootstrap_blender.py: Blender for Finish on Linux, from the viewer.

Finish needs Blender, and on a Linux server installing it meant leaving the viewer to
download and unpack blender.org's tarball by hand (done that way on an NVIDIA pod test,
2026-10-01: it runs headless with no extra libraries).
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import bootstrap_blender as boot  # noqa: E402

LISTING = """
<a href="blender-4.2.9-linux-x64.tar.xz">blender-4.2.9-linux-x64.tar.xz</a>
<a href="blender-4.2.23-linux-x64.tar.xz">blender-4.2.23-linux-x64.tar.xz</a>
<a href="blender-4.2.23-windows-x64.zip">blender-4.2.23-windows-x64.zip</a>
<a href="blender-4.2.23-linux-x64.tar.xz.sha256">...</a>
"""


def test_picks_the_newest_linux_lts_build_by_version_not_by_text():
    assert boot.newest_tarball(LISTING) == "blender-4.2.23-linux-x64.tar.xz"


def test_no_linux_build_is_an_error_not_a_guess():
    with pytest.raises(SystemExit):
        boot.newest_tarball('<a href="blender-4.2.1-windows-x64.zip">x</a>')


def test_installs_where_the_lab_already_looks(tmp_path):
    from image_to_3dlab.blender import candidates

    binary = boot.target(tmp_path) / "blender"
    binary.parent.mkdir()
    binary.write_text("")
    assert binary in candidates("linux", tmp_path)


def test_unpack_renames_the_versioned_folder(tmp_path):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:xz") as tar:
        info = tarfile.TarInfo("blender-4.2.23-linux-x64/blender")
        payload = b"#!/bin/sh\necho blender\n"
        info.size = len(payload)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(payload))
    archive = tmp_path / "b.tar.xz"
    archive.write_bytes(data.getvalue())
    boot.unpack(archive, tmp_path)
    assert (boot.target(tmp_path) / "blender").is_file()
    assert not (tmp_path / "blender-4.2.23-linux-x64").exists()


def test_refuses_anything_but_linux_x86_64():
    assert boot.supported("Linux", "x86_64") is True
    assert boot.supported("Darwin", "arm64") is False
    assert boot.supported("Linux", "aarch64") is False


def test_announcement_names_source_size_and_destination(tmp_path):
    text = boot.announcement("blender-4.2.23-linux-x64.tar.xz", 380 * 1024 ** 2, tmp_path)
    assert "download.blender.org" in text and "380 MB" in text and str(tmp_path) in text
    assert "GPL" in text
