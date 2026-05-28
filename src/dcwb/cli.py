from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from dcwb.calibrate import calibrate_camera, JST
from dcwb.render import render_event, CAMERAS
from dcwb.verify import generate_verify_report
from dcwb import prune as prune_mod

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
    pp.add_argument("--purge", action="store_true", help="Delete trash past the retention window")
    pp.add_argument("--restore", metavar="SEGMENT_ID|all", help="Restore quarantined segment(s)")
    pp.add_argument("--retention-days", type=int, default=None, help="Override retention_days")

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

def _cmd_prune(args) -> int:
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

    candidates = prune_mod.find_candidates(usb_root, cfg, now)
    print(prune_mod.format_report(candidates))

    if args.apply:
        rows = prune_mod.quarantine(usb_root, candidates, cfg, now)
        purged = prune_mod.purge(usb_root, cfg, now)
        print(f"[prune] quarantined {len(rows)} file(s); purged {purged} expired", file=sys.stderr)
    elif args.purge:
        purged = prune_mod.purge(usb_root, cfg, now)
        print(f"[prune] purged {purged} expired file(s)", file=sys.stderr)
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
        "prune-recent": _cmd_prune,
    }[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())
