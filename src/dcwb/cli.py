from __future__ import annotations
import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from dcwb.calibrate import calibrate_camera, JST
from dcwb.render import render_event, CAMERAS
from dcwb.verify import generate_verify_report
from dcwb import prune as prune_mod
from dcwb.highlight import highlight_day
from dcwb.vlm import VlmClient, VlmConfig, VlmUnavailableError

DEFAULT_PROFILES_DIR = Path("profiles")
DEFAULT_OUT_ROOT = Path("/Users/noguchi/AI/dashcamwb/corrected")
DEFAULT_PIPELINE_CFG = Path("pipeline.json")

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dcwb", description="Tesla DashCam White Balance")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate", help="Build per-camera profiles via statistical mining")
    pc.add_argument("--source", type=Path, required=True,
                    help="Path to QNAP root containing SentryClips/RecentClips/SavedClips")
    pc.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    pc.add_argument("--max-samples-per-event", type=int, default=3)

    pr = sub.add_parser("render", help="Render one event")
    pr.add_argument("event_dir", type=Path)
    pr.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    pr.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    pr.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CFG)
    pr.add_argument("--encoder", default="h264_videotoolbox")
    pr.add_argument("--bitrate-kbps", type=int, default=12000)

    pv = sub.add_parser("verify", help="Generate HTML before/after report")
    pv.add_argument("event_dir", type=Path)
    pv.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    pv.add_argument("--out-html", type=Path, default=Path("verify.html"))
    pv.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CFG)

    pa = sub.add_parser("render-all", help="Render every event in a source directory")
    pa.add_argument("--source", type=Path, required=True,
                    help="Directory containing event subdirectories (e.g. SentryClips)")
    pa.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    pa.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    pa.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CFG)
    pa.add_argument("--encoder", default="h264_videotoolbox")
    pa.add_argument("--bitrate-kbps", type=int, default=12000)

    ps = sub.add_parser("serve", help="Local browser UI for /Volumes/sentryusb")
    ps.add_argument("--source", type=Path, default=Path("/Volumes/sentryusb"))
    ps.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    ps.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ps.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CFG)
    ps.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8765)
    ps.add_argument("--debug", action="store_true")

    pp = sub.add_parser("prune-recent", help="Quarantine low-motion RecentClips (dry-run by default)")
    pp.add_argument("--source", type=Path, default=Path("/Volumes/sentryusb"))
    pp.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CFG)
    pp.add_argument("--apply", action="store_true", help="Quarantine candidates (then purge expired)")
    pp.add_argument("--purge", action="store_true", help="Delete trash past the retention window (no extra effect when --apply is given)")
    pp.add_argument("--restore", metavar="SEGMENT_ID|all", help="Restore quarantined segment(s)")
    pp.add_argument("--retention-days", type=int, default=None, help="Override retention_days")

    ph = sub.add_parser("highlight-day", help="Create a daily front-camera drive highlight")
    ph.add_argument("--source", type=Path, default=Path("/Volumes/sentryusb"))
    ph.add_argument("--date", required=True, help="RecentClips date directory, e.g. 2026-05-08")
    ph.add_argument("--out-root", type=Path, default=Path("highlights"))
    ph.add_argument("--style", choices=("fast", "cruise"), default="fast")
    ph.add_argument("--allow-no-sei", action="store_true")
    ph.add_argument("--encoder", default="h264_videotoolbox")
    ph.add_argument("--bitrate-kbps", type=int, default=12000)
    ph.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CFG)
    ph.add_argument("--vlm-endpoint", default=None, help="Override highlight_ai.endpoint")
    ph.add_argument("--vlm-model", default=None, help="Override highlight_ai.model")
    ph.add_argument("--allow-no-ai", action="store_true", help="Fall back to the MVP scorer if the VLM is unavailable")
    ph.add_argument("--no-vlm-cache", action="store_true", help="Ignore and overwrite the per-day VLM cache")
    ph.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR, help="Camera profiles dir (front.json) for white balance")
    ph.add_argument("--no-white-balance", action="store_true", help="Do not apply A x B white-balance correction to excerpts")
    ph.add_argument("--no-look", action="store_true", help="Do not apply the creative look grade (S-curve + saturation) to excerpts")

    return p

def _cmd_calibrate(args) -> int:
    args.profiles_dir.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        print(f"[calibrate] {cam} ...", file=sys.stderr)
        prof = calibrate_camera(
            camera=cam,
            source_root=args.source,
            max_per_event=args.max_samples_per_event,
        )
        prof.to_json(args.profiles_dir / f"{cam}.json")
        print(
            f"[calibrate] {cam}: gain_r={prof.gain_r:.3f} gain_b={prof.gain_b:.3f} "
            f"samples={prof.calibration.samples_used}",
            file=sys.stderr,
        )
    return 0

def _cmd_render(args) -> int:
    cfg = json.loads(args.pipeline_config.read_text())
    render_event(
        event_dir=args.event_dir,
        out_root=args.out_root,
        profiles_dir=args.profiles_dir,
        pipeline_cfg=cfg,
        encoder=args.encoder,
        bitrate_kbps=args.bitrate_kbps,
    )
    print(f"[render] {args.event_dir.name} → {args.out_root / args.event_dir.name}", file=sys.stderr)
    return 0

def _cmd_verify(args) -> int:
    pipeline_cfg = None
    if args.pipeline_config is not None and args.pipeline_config.exists():
        pipeline_cfg = json.loads(args.pipeline_config.read_text())
    generate_verify_report(
        event_dir=args.event_dir,
        profiles_dir=args.profiles_dir,
        out_html=args.out_html,
        pipeline_cfg=pipeline_cfg,
    )
    print(f"[verify] wrote {args.out_html}", file=sys.stderr)
    return 0

def _cmd_serve(args) -> int:
    from dcwb.serve.app import create_app
    cfg = json.loads(args.pipeline_config.read_text()) if args.pipeline_config.exists() else {}
    # Resolve to absolute paths: Flask's send_file() resolves relative paths
    # against app.root_path (the package dir), not CWD, which would cause
    # cache PNGs and corrected mp4s to 404 even when present.
    app = create_app(
        usb_root=args.source.resolve(),
        profiles_dir=args.profiles_dir.resolve(),
        out_root=args.out_root.resolve(),
        pipeline_cfg=cfg,
        cache_root=args.cache_dir.resolve(),
    )
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0

def _cmd_render_all(args) -> int:
    cfg = json.loads(args.pipeline_config.read_text())
    events = sorted(p for p in args.source.iterdir() if p.is_dir())
    for ev in events:
        print(f"[render-all] {ev.name}", file=sys.stderr)
        try:
            render_event(
                event_dir=ev,
                out_root=args.out_root,
                profiles_dir=args.profiles_dir,
                pipeline_cfg=cfg,
                encoder=args.encoder,
                bitrate_kbps=args.bitrate_kbps,
            )
        except Exception as e:
            print(f"[render-all] FAILED {ev.name}: {e}", file=sys.stderr)
    return 0


def _print_prune_progress(label: str, current: int, total: int, unit: str, **counts: int) -> None:
    pct = 100.0 if total == 0 else current * 100.0 / total
    suffix = "".join(f" {name}={value}" for name, value in counts.items())
    print(f"[prune] {label}: {current}/{total} {unit} {pct:.1f}%{suffix}", file=sys.stderr, flush=True)


def _cmd_prune_recent(args) -> int:
    if args.restore and (args.apply or args.purge):
        print("[prune] --restore cannot be combined with --apply/--purge", file=sys.stderr)
        return 1
    usb_root = args.source.resolve()
    full = json.loads(args.pipeline_config.read_text()) if args.pipeline_config.exists() else {}
    cfg = {**prune_mod.DEFAULT_PRUNE_CFG, **full.get("prune", {})}
    if args.retention_days is not None:
        cfg["retention_days"] = args.retention_days
    now = datetime.now(JST)

    if args.restore:
        n = prune_mod.restore(usb_root, cfg, args.restore)
        print(f"[prune] restored {n} file(s)", file=sys.stderr)
        return 0

    candidates = prune_mod.find_candidates(
        usb_root,
        cfg,
        now,
        progress=lambda current, total, candidate_count: _print_prune_progress(
            "scanning RecentClips",
            current,
            total,
            "segment(s)",
            candidates=candidate_count,
        ),
    )
    print(prune_mod.format_report(candidates))

    if args.apply:
        rows = prune_mod.quarantine(
            usb_root,
            candidates,
            cfg,
            now,
            progress=lambda current, total, moved: _print_prune_progress(
                "quarantining",
                current,
                total,
                "file(s)",
                moved=moved,
            ),
        )
        purged = prune_mod.purge(usb_root, cfg, now)
        print(f"[prune] quarantined {len(rows)} file(s); purged {purged} expired", file=sys.stderr)
    elif args.purge:
        purged = prune_mod.purge(usb_root, cfg, now)
        print(f"[prune] purged {purged} expired file(s)", file=sys.stderr)
    return 0


def _cmd_highlight_day(args) -> int:
    cfg_all = (
        json.loads(args.pipeline_config.read_text())
        if args.pipeline_config.exists()
        else {}
    )
    vlm_cfg = VlmConfig.from_dict(cfg_all.get("highlight_ai"))
    if args.vlm_endpoint:
        vlm_cfg = replace(vlm_cfg, endpoint=args.vlm_endpoint)
    if args.vlm_model:
        vlm_cfg = replace(vlm_cfg, model=args.vlm_model)

    client = VlmClient(vlm_cfg)
    selection = "mvp"
    try:
        client.health_check()
    except VlmUnavailableError as e:
        if not args.allow_no_ai:
            print(f"[highlight] VLM unavailable: {e}; pass --allow-no-ai to use the MVP scorer", file=sys.stderr)
            return 1
        print(f"[highlight] VLM unavailable: {e}; falling back to MVP scorer", file=sys.stderr)
        client = None
        selection = "mvp-fallback"

    def on_progress(phase: str, done: int, total: int, note: str) -> None:
        # Telemetry scans hundreds of clips; throttle to every 25 and the last.
        # VLM calls are few and slow, so report each one.
        if phase == "telemetry" and done != total and done % 25 != 0:
            return
        print(f"[highlight] {phase} {done}/{total} {note}".rstrip(), file=sys.stderr, flush=True)

    try:
        result = highlight_day(
            source_root=args.source.resolve(),
            date=args.date,
            out_root=args.out_root.resolve(),
            style=args.style,
            allow_no_sei=args.allow_no_sei,
            encoder=args.encoder,
            bitrate_kbps=args.bitrate_kbps,
            vlm_client=client,
            use_cache=not args.no_vlm_cache,
            selection=selection,
            on_progress=on_progress,
            profiles_dir=args.profiles_dir.resolve(),
            awb_cfg=cfg_all.get("awb"),
            white_balance=not args.no_white_balance,
            look_cfg=cfg_all.get("look"),
            apply_look=not args.no_look,
        )
    except FileNotFoundError as e:
        print(f"[highlight] {e}", file=sys.stderr)
        return 1
    if result.excerpt_count == 0:
        print("[highlight] no eligible clips; wrote manifest only", file=sys.stderr)
        print(f"[highlight] manifest {result.manifest_path}", file=sys.stderr)
        return 0
    print(f"[highlight] wrote {result.output_path}", file=sys.stderr)
    print(f"[highlight] manifest {result.manifest_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return {
        "calibrate": _cmd_calibrate,
        "render": _cmd_render,
        "verify": _cmd_verify,
        "render-all": _cmd_render_all,
        "serve": _cmd_serve,
        "prune-recent": _cmd_prune_recent,
        "highlight-day": _cmd_highlight_day,
    }[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())
