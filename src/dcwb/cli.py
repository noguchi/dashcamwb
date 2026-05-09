from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from dcwb.calibrate import calibrate_camera
from dcwb.render import render_event, CAMERAS
from dcwb.verify import generate_verify_report

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

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return {
        "calibrate": _cmd_calibrate,
        "render": _cmd_render,
        "verify": _cmd_verify,
        "render-all": _cmd_render_all,
    }[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())
