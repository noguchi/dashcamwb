from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
from dcwb.profile import Profile, CalibrationMeta
from dcwb.serve.app import create_app
from tests.fixtures.make_synthetic import make_event

CAMERAS = ("front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater")


@pytest.fixture
def app(tmp_path):
    usb = tmp_path / "usb"
    (usb / "SentryClips" / "2026-05-05_13-50-46").mkdir(parents=True)
    # one fully-formed synthetic event
    make_event(usb / "SentryClips" / "2026-05-05_13-50-46")

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    for cam in CAMERAS:
        cast = (1.1, 1.0, 0.9) if cam == "front" else (1.0, 1.0, 1.0)
        p = Profile.from_white_point(
            cam, np.array(cast) * 200.0,
            CalibrationMeta(samples_used=10, events_sampled=2, method="t",
                            calibrated_at=datetime.now(timezone.utc),
                            samples_per_event_max=3),
        )
        p.to_json(profiles_dir / f"{cam}.json")

    out_root = tmp_path / "corrected"
    cache_root = tmp_path / "cache"
    return create_app(
        usb_root=usb,
        profiles_dir=profiles_dir,
        out_root=out_root,
        pipeline_cfg={"awb": {
            "samples_per_clip": 3, "minkowski_p": 6,
            "saturation_high": 0.97, "saturation_low": 0.03,
            "gain_min": 0.7, "gain_max": 1.5, "night_attenuation": 0.5,
        }},
        cache_root=cache_root,
    )


def test_root_lists_sources(app):
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    body = r.data.decode()
    assert "SentryClips" in body
    assert "SavedClips" in body
    assert "RecentClips" in body


def test_events_route(app):
    client = app.test_client()
    r = client.get("/s/SentryClips")
    assert r.status_code == 200
    assert "2026-05-05_13-50-46" in r.data.decode()


def test_event_detail_route(app):
    client = app.test_client()
    r = client.get("/s/SentryClips/2026-05-05_13-50-46/")
    assert r.status_code == 200
    body = r.data.decode()
    for cam in CAMERAS:
        assert cam in body


def test_preview_png_route(app):
    client = app.test_client()
    r = client.get("/preview/SentryClips/2026-05-05_13-50-46/front/before.png")
    assert r.status_code == 200
    assert r.mimetype == "image/png"
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_reindex_redirects(app):
    client = app.test_client()
    r = client.post("/reindex")
    assert r.status_code == 303
    assert r.headers["Location"].endswith("/")


def test_render_endpoint_redirects_and_creates_job(app):
    client = app.test_client()
    # short-circuit render by pre-creating the corrected/<event>/ dir
    out_root = app.config["DCWB_OUT_ROOT"]
    rendered = out_root / "2026-05-05_13-50-46"
    rendered.mkdir(parents=True)
    for cam in CAMERAS:
        (rendered / f"2026-05-05_13-49-39-{cam}.mp4").write_bytes(b"x")
    r = client.post("/render/SentryClips/2026-05-05_13-50-46")
    assert r.status_code == 303
    assert "/s/SentryClips/2026-05-05_13-50-46/" in r.headers["Location"]


def test_jobs_status_json(app, monkeypatch):
    # call render then immediately query status
    client = app.test_client()
    out_root = app.config["DCWB_OUT_ROOT"]
    rendered = out_root / "2026-05-05_13-50-46"
    rendered.mkdir(parents=True)
    for cam in CAMERAS:
        (rendered / f"2026-05-05_13-49-39-{cam}.mp4").write_bytes(b"x")
    r = client.post("/render/SentryClips/2026-05-05_13-50-46")
    # extract job_id from Location query param
    loc = r.headers["Location"]
    assert "job=" in loc
    job_id = loc.split("job=")[-1]
    js = client.get(f"/jobs/{job_id}")
    assert js.status_code == 200
    payload = js.get_json()
    assert payload["status"] in ("done", "queued", "running")


def test_corrected_route_rejects_event_escape(app):
    """Regression: /corrected/<src>/<event>/<filename> must be bounded by the
    event directory. A traversal that escapes <event> but stays inside
    out_root previously returned 200 with a sibling event's mp4."""
    out_root = app.config["DCWB_OUT_ROOT"]
    (out_root / "A").mkdir(parents=True)
    (out_root / "B").mkdir(parents=True)
    secret = b"this should not leak"
    (out_root / "B" / "secret.mp4").write_bytes(secret)
    client = app.test_client()
    for path in (
        "/corrected/SentryClips/A/%2e%2e/B/secret.mp4",
        "/corrected/SentryClips/A/..%2fB%2fsecret.mp4",
        "/corrected/SentryClips/A/%2e%2e%2fB%2fsecret.mp4",
    ):
        r = client.get(path)
        assert not (r.status_code == 200 and r.data == secret), \
            f"path traversal escape succeeded via {path}"


def test_corrected_video_range(app):
    out_root = app.config["DCWB_OUT_ROOT"]
    rendered = out_root / "2026-05-05_13-50-46"
    rendered.mkdir(parents=True)
    payload = bytes(range(256)) * 16   # 4096 bytes
    (rendered / "2026-05-05_13-49-39-front.mp4").write_bytes(payload)
    client = app.test_client()
    r = client.get(
        "/corrected/SentryClips/2026-05-05_13-50-46/2026-05-05_13-49-39-front.mp4",
        headers={"Range": "bytes=0-15"},
    )
    assert r.status_code == 206
    assert r.headers["Content-Range"].startswith("bytes 0-15/")
    assert len(r.data) == 16


def test_usb_missing_shows_banner(tmp_path):
    app = create_app(
        usb_root=tmp_path / "does-not-exist",
        profiles_dir=tmp_path / "profiles",
        out_root=tmp_path / "out",
        pipeline_cfg={"awb": {}},
        cache_root=tmp_path / "cache",
    )
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert "見つかりません" in r.data.decode() or "not found" in r.data.decode().lower()
