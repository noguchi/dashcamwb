from __future__ import annotations
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from dcwb.render import render_event
from .index import Event

Status = Literal["queued", "running", "done", "failed"]


@dataclass
class JobState:
    id: str
    source: str
    event_name: str
    status: Status = "queued"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


def _is_already_rendered(out_root: Path, event: Event) -> bool:
    target = out_root / event.name
    if not target.exists():
        return False
    expected = {clip.name for clip in event.clips}
    if not expected:
        return False
    existing = {f.name for f in target.glob("*.mp4")}
    return expected.issubset(existing)


class JobQueue:
    def __init__(
        self,
        out_root: Path,
        profiles_dir: Path,
        pipeline_cfg: dict,
        encoder: str = "h264_videotoolbox",
        bitrate_kbps: int = 12000,
    ) -> None:
        self.out_root = out_root
        self.profiles_dir = profiles_dir
        self.pipeline_cfg = pipeline_cfg
        self.encoder = encoder
        self.bitrate_kbps = bitrate_kbps
        self._states: dict[str, JobState] = {}
        self._event_to_job: dict[tuple[str, str], str] = {}  # (source, name) -> active job_id
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1)

    def enqueue(self, event: Event) -> str:
        with self._lock:
            if _is_already_rendered(self.out_root, event):
                jid = str(uuid.uuid4())
                self._states[jid] = JobState(
                    id=jid, source=event.source, event_name=event.name,
                    status="done", started_at=datetime.now(), finished_at=datetime.now(),
                )
                return jid
            key = (event.source, event.name)
            existing = self._event_to_job.get(key)
            if existing is not None:
                state = self._states.get(existing)
                if state is not None and state.status in ("queued", "running"):
                    return existing
            jid = str(uuid.uuid4())
            self._states[jid] = JobState(id=jid, source=event.source, event_name=event.name)
            self._event_to_job[key] = jid
        self._pool.submit(self._run, jid, event)
        return jid

    def get(self, job_id: str) -> JobState:
        with self._lock:
            return self._states[job_id]

    def _run(self, jid: str, event: Event) -> None:
        with self._lock:
            state = self._states[jid]
            state.status = "running"
            state.started_at = datetime.now()
        error: str | None = None
        try:
            if event.source == "RecentClips":
                # symlink only the subset clips into a temp dir so render_event's
                # glob("*.mp4") picks up just the pseudo-event range.
                with tempfile.TemporaryDirectory(prefix=f"dcwb-render-{event.name}-") as tmp:
                    tmp_dir = Path(tmp) / event.name
                    tmp_dir.mkdir()
                    for clip in event.clips:
                        (tmp_dir / clip.name).symlink_to(clip)
                    render_event(
                        event_dir=tmp_dir,
                        out_root=self.out_root,
                        profiles_dir=self.profiles_dir,
                        pipeline_cfg=self.pipeline_cfg,
                        encoder=self.encoder,
                        bitrate_kbps=self.bitrate_kbps,
                    )
            else:
                render_event(
                    event_dir=event.path,
                    out_root=self.out_root,
                    profiles_dir=self.profiles_dir,
                    pipeline_cfg=self.pipeline_cfg,
                    encoder=self.encoder,
                    bitrate_kbps=self.bitrate_kbps,
                )
        except Exception as e:
            error = str(e)
        with self._lock:
            state.status = "failed" if error else "done"
            state.error = error
            state.finished_at = datetime.now()
