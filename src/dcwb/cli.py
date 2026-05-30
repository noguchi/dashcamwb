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
    ps.add_argument("--host", default="0.0.0.0",
                    help="Bind address; default 0.0.0.0 exposes the UI on the LAN "
                         "(use --host 127.0.0.1 for loopback-only)")
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

    py = sub.add_parser("sync-insta360",
                        help="Sync an Insta360 ride-view with Tesla front clips")
    py.add_argument("insv", nargs="+", type=Path)
    py.add_argument("--recent", required=True, help="RecentClips date dir, YYYY-MM-DD")
    py.add_argument("--insta-flat", type=Path, default=None,
                    help="Reframed flat ride-view mp4 (else v360 auto-reframe)")
    py.add_argument("--source", type=Path, default=Path("/mnt/sentryusb"))
    py.add_argument("--out-root", type=Path, default=Path("sync-work"))
    py.add_argument("--encoder", default="h264_videotoolbox")
    py.add_argument("--bitrate-kbps", type=int, default=12000)
    py.add_argument("--max-duration", type=float, default=None,
                    help="Cap the rendered/correlated window length (seconds) for speed")
    # v360 auto-reframe orientation (only used when --insta-flat is omitted).
    # Defaults suit a forward, right-side-up ride-view on this mount
    # (yaw=180 faces forward away from the rear lens; roll=180 corrects an
    # upside-down-mounted camera).
    py.add_argument("--insta-yaw", type=float, default=180.0)
    py.add_argument("--insta-pitch", type=float, default=12.0,
                    help="Downward tilt; ~12 frames from the rearview-mirror top "
                         "to the center-screen bottom")
    py.add_argument("--insta-roll", type=float, default=180.0)
    py.add_argument("--insta-hfov", type=float, default=82.0)
    py.add_argument("--insta-vfov", type=float, default=52.0)
    py.add_argument("--start-offset", type=float, default=0.0,
                    help="Begin the rendered window this many seconds into the "
                         "Insta360 recording (skip parked footage at the start)")
    py.add_argument("--visual-offset", type=float, default=0.0,
                    help="Extra seconds added to the Tesla lead to correct the "
                         "creation_time anchor (Insta video v -> Tesla epoch v+offset)")

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


def _detect_visual_offset(insv, tesla_cat, anchor_tesla_lead, insv_total,
                          fps=4.0, win=240.0, search=60.0):
    """Time-align the two videos by WHOLE-FRAME MOTION only (no telemetry/IMU).

    Builds the frame-difference envelope (a parked->moving onset signal) of the
    Insta forward fisheye and the Tesla front camera over the start-of-drive
    region and cross-correlates them. Returns ``(offset_s, peak)`` where a
    physical instant at Insta video time ``v`` occurs at Tesla epoch ``v+offset``
    (epoch = Insta creation_time). ``peak`` is the normalized correlation score.
    """
    import numpy as np
    from dcwb.ffmpeg_wrap import frame_diff_envelope
    from dcwb import sync as S

    win = float(min(win, insv_total))
    ti, di = frame_diff_envelope(insv, fps=fps, start=0.0, duration=win,
                                 stream="0:v:0")
    tdur = win + anchor_tesla_lead + search + 10.0
    tc, dc = frame_diff_envelope(tesla_cat, fps=fps, start=0.0, duration=tdur,
                                 stream="0:v:0")
    if len(di) < 10 or len(dc) < 10:
        return 0.0, 0.0
    te = tc - anchor_tesla_lead                       # Tesla epoch time
    ai = np.log1p(np.clip(di, 0.0, None))             # compress spikes
    at = np.log1p(np.clip(dc, 0.0, None))
    g = np.arange(0.0, max(win - 5.0, 1.0), 1.0 / fps)
    aii = np.interp(g, ti, ai, left=0.0, right=0.0)
    ati = np.interp(g, te, at, left=0.0, right=0.0)
    lag, peak = S.normalized_xcorr(aii, ati, max_lag=int(search * fps))
    return lag / fps, float(peak)


def run_sync_insta360(*, insv, recent, insta_flat, source, out_root,
                      encoder, bitrate_kbps, max_duration=None,
                      insta_yaw=180.0, insta_pitch=12.0, insta_roll=180.0,
                      insta_hfov=82.0, insta_vfov=52.0, start_offset=0.0,
                      visual_offset=0.0) -> int:
    """Orchestrate Insta360<->Tesla sync. A coarse anchor comes from file
    timestamps (Insta creation_time vs Tesla front filenames); the fine
    alignment is PURELY VISUAL — the moment the car starts moving is detected by
    whole-frame motion (frame differencing) in both videos and cross-correlated
    (no accelerometer/gyro/GPS/speed used for sync). Renders a side-by-side
    combined.mp4 with a Tesla telemetry overlay + a manifest for the web player."""
    import numpy as np
    from datetime import timedelta
    from dcwb import insta360
    from dcwb import sync as S
    from dcwb.telemetry import iter_segment_frames
    from dcwb.ffmpeg_wrap import (probe_duration, concat_clips, render_sidebyside,
                                  reframe_insv)

    insv = [Path(p) for p in insv]
    out_root = Path(out_root)
    rate = 10.0

    # --- absolute window from insv ---
    start_jst = insta360.to_jst(insta360.read_creation_time(insv[0]))
    insv_total = sum(probe_duration(p) for p in insv)
    end_jst = start_jst + timedelta(seconds=insv_total)
    print(f"[sync] insta start {start_jst.isoformat()} dur {insv_total:.0f}s", file=sys.stderr)

    # --- Tesla front clips overlapping the window ---
    day_dir = Path(source) / "RecentClips" / recent
    fronts = S.select_front_clips(day_dir, start_jst, end_jst)
    if not fronts:
        print(f"[sync] no Tesla front clips overlap in {day_dir}", file=sys.stderr)
        return 1
    print(f"[sync] {len(fronts)} Tesla front clips", file=sys.stderr)

    # --- Tesla SEI samples for the telemetry overlay ONLY (not used for sync) ---
    t_abs, speed, steer, gear = [], [], [], []
    for fp in fronts:
        cs = S._front_start(fp.name)
        frames = list(iter_segment_frames(fp))
        if not frames:
            continue
        dur = probe_duration(fp)
        fps = len(frames) / dur if dur > 0 else 36.0
        base = (cs - start_jst).total_seconds()
        for f in frames:
            t_abs.append(base + f.frame_index / fps)
            speed.append(f.speed_mps); steer.append(f.steering_deg)
            gear.append(f.gear)
    t_abs = np.asarray(t_abs)
    # np.interp needs strictly-increasing time; overlapping front-clip boundaries
    # can break monotonicity, so sort + dedup all parallel arrays together.
    order = np.argsort(t_abs, kind="stable")
    t_abs = t_abs[order]
    speed = np.asarray(speed)[order]; steer = np.asarray(steer)[order]
    gear = np.asarray(gear)[order]
    keep = np.concatenate(([True], np.diff(t_abs) > 1e-6))
    t_abs = t_abs[keep]; speed = speed[keep]; steer = steer[keep]; gear = gear[keep]
    gt, speed_u = S.resample_uniform(t_abs, speed, rate)
    _, steer_u = S.resample_uniform(t_abs, steer, rate)
    # Gear is categorical — map each grid point to the nearest real sample.
    gear_idx = np.clip(np.searchsorted(t_abs, gt), 0, len(gear) - 1)

    # --- coarse anchor from file timestamps only (no motion data) ---
    # tesla_concat local t=0 is fronts[0] start; insta local t=0 is start_jst.
    tesla0 = (S._front_start(fronts[0].name) - start_jst).total_seconds()  # ~ -53
    anchor_tesla_lead = -tesla0                                            # ~ +53

    out_dir = out_root / "sync" / recent
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = encoder
    cap = max_duration if max_duration else min(insv_total, 600.0)
    start_offset = max(0.0, float(start_offset))

    # --- Tesla concat (render once; covers visual detection + final window) ---
    tesla_cat = out_dir / "tesla-concat.mp4"
    cover = start_offset + anchor_tesla_lead + cap + 90.0
    n_clips = min(len(fronts), max(1, int(cover // 60) + 2))
    concat_clips(fronts[:n_clips], tesla_cat, encoder=enc, bitrate_kbps=bitrate_kbps)

    # --- VISUAL fine-alignment: detect the moment the car starts moving in both
    # videos (whole-frame motion / frame differencing) and align by it. No
    # accelerometer/gyro/GPS/speed is used for sync. --visual-offset overrides. ---
    if visual_offset:
        vis_off, vis_peak, method = float(visual_offset), float("nan"), "manual"
    else:
        vis_off, vis_peak = _detect_visual_offset(
            insv[0], tesla_cat, anchor_tesla_lead, insv_total)
        method = "visual"
        if vis_peak < 0.30:        # weak lock -> keep the bare timestamp anchor
            print(f"[sync] WARNING weak visual lock (peak={vis_peak:.3f}); "
                  f"using timestamp anchor only", file=sys.stderr)
            vis_off, method = 0.0, "anchor(weak-visual)"
    tesla_lead = anchor_tesla_lead + vis_off
    print(f"[sync] anchor_tesla_lead={anchor_tesla_lead:.1f}s visual_offset={vis_off:.1f}s "
          f"(peak={vis_peak:.3f}, {method}) -> tesla_lead={tesla_lead:.1f}s", file=sys.stderr)

    # --- alignment trims. At display-local L the Insta panel shows epoch
    # (start_offset + L); the same instant is Tesla epoch (start_offset+L+vis_off).
    # Tesla concat local 0 = epoch -anchor_tesla_lead, so trim Tesla by
    # (start_offset + tesla_lead) and Insta by 0. ---
    right_start = start_offset + tesla_lead
    left_start = 0.0
    if right_start < 0:
        left_start = -right_start
        right_start = 0.0
    print(f"[sync] start_offset={start_offset:.1f}s left_start={left_start:.3f} "
          f"right_start={right_start:.3f}", file=sys.stderr)
    # Player seeks tesla.currentTime = insta.currentTime + delta = right_start.
    res = S.SyncResult(delta_s=right_start, confidence=float(vis_peak),
                       signal="visual-motion", anchor_guess=anchor_tesla_lead)

    # --- Insta display: provided flat export, else v360 reframe of the window ---
    if insta_flat:
        display = Path(insta_flat)
    else:
        display = out_dir / "insta-flat.mp4"
        reframe_insv(insv[0], display, yaw=insta_yaw, pitch=insta_pitch,
                     roll=insta_roll, h_fov=insta_hfov, v_fov=insta_vfov,
                     out_w=1280, out_h=720,
                     encoder=enc, bitrate_kbps=bitrate_kbps,
                     start=start_offset, duration=cap)

    # --- telemetry overlay (~1 Hz). The Tesla panel at display-local L shows
    # epoch (start_offset + vis_off + L), so label by
    # local = gt[k] - start_offset - vis_off. ---
    rows = []
    step = max(1, int(rate))
    for k in range(0, len(gt), step):
        local = gt[k] - start_offset - vis_off
        if local < 0:
            continue
        if local > cap:
            break
        rows.append((float(local), float(speed_u[k]), float(steer_u[k]),
                     str(gear[gear_idx[k]])))
    ass = out_dir / "telemetry.ass"
    ass.write_text(S.telemetry_ass(rows, play_w=2560, play_h=720))

    combined = out_dir / f"combined-{recent}.mp4"
    render_sidebyside(display, tesla_cat, combined,
                      left_start=left_start, right_start=right_start,
                      duration=cap, ass_path=ass, encoder=enc,
                      bitrate_kbps=bitrate_kbps, panel_h=720)

    manifest = S.write_sync_manifest(
        out_dir, res, insta_display=str(display), tesla_concat=str(tesla_cat),
        combined=str(combined), date=recent, telemetry=rows)
    print(f"[sync] wrote {combined}", file=sys.stderr)
    print(f"[sync] manifest {manifest}", file=sys.stderr)
    return 0


def _cmd_sync_insta360(args) -> int:
    return run_sync_insta360(
        insv=args.insv, recent=args.recent, insta_flat=args.insta_flat,
        source=args.source, out_root=args.out_root,
        encoder=args.encoder, bitrate_kbps=args.bitrate_kbps,
        max_duration=args.max_duration,
        insta_yaw=args.insta_yaw, insta_pitch=args.insta_pitch,
        insta_roll=args.insta_roll, insta_hfov=args.insta_hfov,
        insta_vfov=args.insta_vfov, start_offset=args.start_offset,
        visual_offset=args.visual_offset)


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
        "sync-insta360": _cmd_sync_insta360,
    }[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())
