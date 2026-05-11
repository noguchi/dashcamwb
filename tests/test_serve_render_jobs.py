from __future__ import annotations
import time
from datetime import datetime
from pathlib import Path
import pytest
from dcwb.serve.index import Event, CAMERAS
from dcwb.serve import render_jobs


def _event(tmp_path: Path, name: str = "evt", source: str = "SentryClips") -> Event:
    ev_dir = tmp_path / source / name
    ev_dir.mkdir(parents=True)
    clips: list[Path] = []
    for cam in CAMERAS:
        c = ev_dir / f"2026-05-05_13-49-39-{cam}.mp4"
        c.write_bytes(b"")
        clips.append(c)
    return Event(
        source=source, name=name, path=ev_dir, clips=clips,
        start=datetime(2026, 5, 5, 13, 49, 39),
        end=datetime(2026, 5, 5, 13, 50, 39),
    )


def test_enqueue_short_circuits_when_already_rendered(tmp_path):
    ev = _event(tmp_path)
    out_root = tmp_path / "corrected"
    rendered = out_root / ev.name
    rendered.mkdir(parents=True)
    for cam in CAMERAS:
        (rendered / f"2026-05-05_13-49-39-{cam}.mp4").write_bytes(b"")
    queue = render_jobs.JobQueue(out_root=out_root, profiles_dir=tmp_path / "p", pipeline_cfg={})
    job_id = queue.enqueue(ev)
    state = queue.get(job_id)
    assert state.status == "done"
    assert state.error is None


def test_is_already_rendered_requires_every_clip_for_multi_segment(tmp_path):
    """Regression: a multi-segment event must not be considered fully rendered
    until every expected output mp4 exists (one per source clip). The previous
    check accepted any single file per camera and silently dropped the trailing
    segments from the UI."""
    ev_dir = tmp_path / "SentryClips" / "evt"
    ev_dir.mkdir(parents=True)
    clips: list[Path] = []
    for ts in ("2026-05-05_13-49-39", "2026-05-05_13-50-39"):
        for cam in CAMERAS:
            c = ev_dir / f"{ts}-{cam}.mp4"
            c.write_bytes(b"")
            clips.append(c)
    ev = Event(
        source="SentryClips", name="evt", path=ev_dir, clips=clips,
        start=datetime(2026, 5, 5, 13, 49, 39),
        end=datetime(2026, 5, 5, 13, 51, 39),
    )
    out_root = tmp_path / "corrected"
    rendered = out_root / ev.name
    rendered.mkdir(parents=True)
    for cam in CAMERAS:
        (rendered / f"2026-05-05_13-49-39-{cam}.mp4").write_bytes(b"x")
    assert render_jobs._is_already_rendered(out_root, ev) is False
    for cam in CAMERAS:
        (rendered / f"2026-05-05_13-50-39-{cam}.mp4").write_bytes(b"x")
    assert render_jobs._is_already_rendered(out_root, ev) is True


def test_enqueue_invokes_render_event_when_missing(tmp_path, monkeypatch):
    ev = _event(tmp_path)
    out_root = tmp_path / "corrected"
    captured: dict = {}

    def fake_render(event_dir: Path, out_root: Path, profiles_dir: Path,
                    pipeline_cfg: dict, encoder: str = "h264_videotoolbox",
                    bitrate_kbps: int = 12000) -> None:
        captured["event_dir"] = event_dir
        captured["out_root"] = out_root
        (out_root / event_dir.name).mkdir(parents=True, exist_ok=True)
        for cam in CAMERAS:
            (out_root / event_dir.name / f"2026-05-05_13-49-39-{cam}.mp4").write_bytes(b"x")

    monkeypatch.setattr(render_jobs, "render_event", fake_render)
    queue = render_jobs.JobQueue(out_root=out_root, profiles_dir=tmp_path / "p", pipeline_cfg={"awb": {}})
    job_id = queue.enqueue(ev)
    # wait up to 2 seconds for serial worker
    for _ in range(40):
        if queue.get(job_id).status in ("done", "failed"):
            break
        time.sleep(0.05)
    assert queue.get(job_id).status == "done"
    assert captured["event_dir"].name == ev.name


def test_enqueue_records_failure_on_exception(tmp_path, monkeypatch):
    ev = _event(tmp_path)
    out_root = tmp_path / "corrected"

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(render_jobs, "render_event", boom)
    queue = render_jobs.JobQueue(out_root=out_root, profiles_dir=tmp_path / "p", pipeline_cfg={"awb": {}})
    job_id = queue.enqueue(ev)
    for _ in range(40):
        if queue.get(job_id).status in ("done", "failed"):
            break
        time.sleep(0.05)
    state = queue.get(job_id)
    assert state.status == "failed"
    assert "boom" in (state.error or "")


def test_recentclips_pseudo_event_uses_tempdir(tmp_path, monkeypatch):
    # simulate RecentClips: day-dir with many clips, event covers a subset
    day = tmp_path / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    subset_clips: list[Path] = []
    for cam in CAMERAS:
        c = day / f"2026-05-08_00-00-00-{cam}.mp4"
        c.write_bytes(b"")
        subset_clips.append(c)
    # an extra clip outside the event range
    (day / f"2026-05-08_03-00-00-front.mp4").write_bytes(b"")
    ev = Event(
        source="RecentClips", name="2026-05-08_0000",
        path=day, clips=subset_clips,
        start=datetime(2026, 5, 8, 0, 0, 0),
        end=datetime(2026, 5, 8, 0, 1, 0),
    )
    out_root = tmp_path / "corrected"
    captured: dict = {}

    def fake_render(event_dir, out_root, profiles_dir, pipeline_cfg, **kw):
        # event_dir should be a temp dir distinct from day-dir
        captured["event_dir"] = event_dir
        captured["mp4s"] = sorted(p.name for p in event_dir.glob("*.mp4"))
        (out_root / ev.name).mkdir(parents=True, exist_ok=True)
        for cam in CAMERAS:
            (out_root / ev.name / f"2026-05-08_00-00-00-{cam}.mp4").write_bytes(b"x")

    monkeypatch.setattr(render_jobs, "render_event", fake_render)
    queue = render_jobs.JobQueue(out_root=out_root, profiles_dir=tmp_path / "p", pipeline_cfg={"awb": {}})
    job_id = queue.enqueue(ev)
    for _ in range(40):
        if queue.get(job_id).status in ("done", "failed"):
            break
        time.sleep(0.05)
    assert queue.get(job_id).status == "done"
    assert captured["event_dir"] != day
    # only the 6 subset clips should be present in the temp dir, not the extra one
    assert len(captured["mp4s"]) == 6
    assert all("00-00-00" in n for n in captured["mp4s"])
