"""scripts/pod_smoke_test.py: the pure parts. Renting a pod is exercised for real only."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pod_smoke_test as smoke
import pytest


def test_the_pod_terminates_itself_even_if_this_machine_dies():
    command = smoke.create_command("NVIDIA A40", "x", 3)
    at = command[command.index("--terminate-after") + 1]
    assert datetime.strptime(at, "%Y-%m-%dT%H:%M:%SZ")
    assert command[command.index("--cloud-type") + 1] == "SECURE"
    assert f"{smoke.PORT}/http" in command[command.index("--ports") + 1]


def test_terminate_after_is_hours_from_now_in_utc():
    now = datetime(2026, 10, 2, 22, 30, tzinfo=timezone.utc)
    assert smoke.terminate_after(3, now) == "2026-10-03T01:30:00Z"


@pytest.mark.parametrize("pod,price", [({"costPerHr": 0.49}, 0.49),
                                       ({"adjustedCostPerHr": 0.5}, 0.5), ({}, None)])
def test_pod_price_reads_what_runpod_bills(pod, price):
    assert smoke.pod_price(pod) == price


def test_ssh_target_parses_runpods_command():
    info = {"podId": "abc", "sshCommand": "ssh root@1.2.3.4 -p 40022 -i ~/.ssh/key"}
    assert smoke.ssh_target(info) == ("root@1.2.3.4", 40022, "~/.ssh/key")
    with pytest.raises(RuntimeError):
        smoke.ssh_target({"podId": "abc"})


@pytest.mark.parametrize("output,expected", [("52428800.000", 52.4288), ("", 0.0),
                                             ("garbage", 0.0)])
def test_curl_speed_is_megabytes_per_second(output, expected):
    assert smoke.mb_per_s(output) == pytest.approx(expected)


def test_install_never_starts_the_lab_in_the_foreground():
    command = smoke.install_command("feat/x")
    assert "--ref feat/x" in command and "--yes" in command and "--no-start" in command
    assert "/feat/x/install.sh" in command  # the branch's installer, not main's


def test_the_lab_listens_for_runpods_proxy():
    command = smoke.lab_command("pod123")
    assert "RUNPOD_POD_ID=pod123" in command and "nohup ./lab" in command
    assert command.rstrip().endswith("&")


def test_multipart_carries_fields_and_files():
    body, ctype = smoke.multipart({"settings": json.dumps({"backend": "trellis"})},
                                  {"image": ("in.png", b"\x89PNG")})
    boundary = ctype.split("boundary=")[1]
    assert body.endswith(f"--{boundary}--\r\n".encode())
    assert b'name="image"; filename="in.png"' in body and b"\x89PNG" in body
    assert b'{"backend": "trellis"}' in body


def test_is_glb_wants_the_magic_and_some_substance():
    assert smoke.is_glb(b"glTF" + b"\0" * 2000)
    assert not smoke.is_glb(b"glTF")
    assert not smoke.is_glb(b"<html>" + b"\0" * 2000)


def test_sse_events_skip_noise():
    text = 'data: {"phase": "queued", "stages": ["photo"]}\n\n: ping\ndata: nope\n'
    assert smoke.sse_events(text) == [{"phase": "queued", "stages": ["photo"]}]


def test_announcement_names_every_route_and_its_size():
    text = smoke.announcement(["trellis", "pixal3d"], ["NVIDIA A40"], 0.8, 3)
    assert "$0.80/hr" in text and "NVIDIA A40" in text and "GB of weights" in text
    assert text.count("GB of weights") == 2


def test_every_route_is_in_the_catalogue():
    import backend_catalog
    assert all(backend_catalog.resolve(r) is not None for r in smoke.ROUTES)


def test_nothing_is_rented_without_a_yes(monkeypatch, tmp_path):
    image = tmp_path / "in.png"
    image.write_bytes(b"x")
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    monkeypatch.setattr(smoke, "rent", lambda *a: pytest.fail("rented"))
    assert smoke.main(["--image", str(image)]) == 1
