from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from flask import (
    Flask, abort, jsonify, render_template, request, send_file, redirect, url_for
)
from .index import scan_sources, Event
from .preview import ensure_previews
from .render_jobs import JobQueue

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def _find_event(events: list[Event], name: str) -> Event | None:
    for ev in events:
        if ev.name == name:
            return ev
    return None


def create_app(
    usb_root: Path,
    profiles_dir: Path,
    out_root: Path,
    pipeline_cfg: dict,
    cache_root: Path,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )
    app.config["DCWB_USB_ROOT"] = usb_root
    app.config["DCWB_PROFILES_DIR"] = profiles_dir
    app.config["DCWB_OUT_ROOT"] = out_root
    app.config["DCWB_PIPELINE_CFG"] = pipeline_cfg
    app.config["DCWB_CACHE_ROOT"] = cache_root

    queue = JobQueue(out_root=out_root, profiles_dir=profiles_dir, pipeline_cfg=pipeline_cfg)
    app.config["DCWB_QUEUE"] = queue

    index_state: dict = {"sources": scan_sources(usb_root)}

    def _sources() -> dict[str, list[Event]]:
        return index_state["sources"]

    @app.route("/")
    def root():
        sources = _sources()
        usb_missing = not usb_root.exists()
        return render_template(
            "sources.html.j2",
            sources=sources,
            usb_root=str(usb_root),
            usb_missing=usb_missing,
        )

    @app.route("/s/<source>")
    def events_view(source: str):
        sources = _sources()
        if source not in sources:
            abort(404)
        events = sources[source]
        rendered_names = set()
        if out_root.exists():
            for child in out_root.iterdir():
                if child.is_dir():
                    rendered_names.add(child.name)
        return render_template(
            "events.html.j2",
            source=source,
            events=events,
            rendered_names=rendered_names,
        )

    @app.route("/s/<source>/<event_name>/")
    def event_detail(source: str, event_name: str):
        sources = _sources()
        if source not in sources:
            abort(404)
        ev = _find_event(sources[source], event_name)
        if ev is None:
            abort(404)
        preview = ensure_previews(ev, profiles_dir, pipeline_cfg, cache_root)
        rendered_dir = out_root / ev.name
        all_cams = ("front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater")
        expected_names = {clip.name for clip in ev.clips}
        existing_names: set[str] = set()
        if rendered_dir.exists():
            existing_names = {
                f.name for f in rendered_dir.glob("*.mp4") if f.name in expected_names
            }
        rendered = bool(expected_names) and expected_names.issubset(existing_names)
        clips_by_cam: dict[str, list[tuple[str, str | None]]] = {cam: [] for cam in all_cams}
        for clip in ev.clips:
            for cam in all_cams:
                if clip.stem.endswith("-" + cam):
                    corrected = clip.name if clip.name in existing_names else None
                    clips_by_cam[cam].append((clip.name, corrected))
                    break
        job_id = request.args.get("job")
        return render_template(
            "event.html.j2",
            source=source, event=ev, preview=preview,
            rendered=rendered, job_id=job_id,
            clips_by_cam=clips_by_cam,
        )

    @app.route("/preview/<source>/<event_name>/<cam>/<kind>.png")
    def preview_png(source: str, event_name: str, cam: str, kind: str):
        if kind not in ("before", "after"):
            abort(404)
        sources = _sources()
        if source not in sources:
            abort(404)
        ev = _find_event(sources[source], event_name)
        if ev is None:
            abort(404)
        preview = ensure_previews(ev, profiles_dir, pipeline_cfg, cache_root)
        if cam not in preview.paths:
            abort(404)
        path = preview.paths[cam][kind]
        return send_file(path, mimetype="image/png", conditional=True)

    @app.route("/render/<source>/<event_name>", methods=["POST"])
    def render(source: str, event_name: str):
        sources = _sources()
        if source not in sources:
            abort(404)
        ev = _find_event(sources[source], event_name)
        if ev is None:
            abort(404)
        jid = queue.enqueue(ev)
        return redirect(url_for("event_detail", source=source, event_name=event_name) + f"?job={jid}", code=303)

    @app.route("/jobs/<job_id>")
    def job_status(job_id: str):
        try:
            state = queue.get(job_id)
        except KeyError:
            abort(404)
        elapsed = None
        if state.started_at is not None:
            end = state.finished_at or datetime.now()
            elapsed = (end - state.started_at).total_seconds()
        return jsonify({
            "id": state.id,
            "status": state.status,
            "error": state.error,
            "elapsed_s": elapsed,
        })

    @app.route("/corrected/<source>/<event_name>/<path:filename>")
    def corrected_file(source: str, event_name: str, filename: str):
        event_root = (out_root / event_name).resolve()
        target = (event_root / filename).resolve()
        if not target.is_relative_to(event_root):
            abort(404)
        if not target.exists():
            abort(404)
        return send_file(target, mimetype="video/mp4", conditional=True)

    @app.route("/reindex", methods=["POST"])
    def reindex():
        index_state["sources"] = scan_sources(usb_root)
        return redirect(url_for("root"), code=303)

    sync_root = (out_root / "sync")

    @app.route("/sync/<date>")
    def sync_player(date):
        return render_template("sync_player.html.j2", date=date)

    @app.route("/sync-data/<date>")
    def sync_data(date):
        f = (sync_root / date / "sync.json").resolve()
        if not f.is_relative_to(sync_root.resolve()) or not f.exists():
            abort(404)
        return send_file(f, mimetype="application/json", conditional=False)

    @app.route("/sync-nudge/<date>", methods=["POST"])
    def sync_nudge(date):
        f = (sync_root / date / "sync.json").resolve()
        if not f.is_relative_to(sync_root.resolve()) or not f.exists():
            abort(404)
        data = json.loads(f.read_text())
        data["delta_s"] = float(request.get_json()["delta_s"])
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return jsonify(ok=True, delta_s=data["delta_s"])

    @app.route("/sync-video/<date>/<path:filename>")
    def sync_video(date, filename):
        base = (sync_root / date).resolve()
        target = (base / filename).resolve()
        if not target.is_relative_to(base) or not target.exists():
            abort(404)
        return send_file(target, mimetype="video/mp4", conditional=True)

    return app
