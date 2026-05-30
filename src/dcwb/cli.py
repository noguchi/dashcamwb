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
    py.add_argument("--insta-pitch", type=float, default=20.0,
                    help="Downward tilt; ~20 frames forward road + the center control panel")
    py.add_argument("--insta-roll", type=float, default=180.0)
    py.add_argument("--insta-hfov", type=float, default=110.0)
    py.add_argument("--insta-vfov", type=float, default=70.0)

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


def run_sync_insta360(*, insv, recent, insta_flat, source, out_root,
                      encoder, bitrate_kbps, max_duration=None,
                      insta_yaw=180.0, insta_pitch=20.0, insta_roll=180.0,
                      insta_hfov=110.0, insta_vfov=70.0) -> int:
    """Orchestrate Insta360<->Tesla sync: anchor by creation_time, refine by
    cross-correlating Tesla yaw-rate against an auto-selected Insta360 gyro axis,
    then render a side-by-side combined.mp4 with a telemetry overlay + manifest."""
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

    # --- Tesla epoch-relative samples (epoch = insta start_jst) ---
    # NOTE: heading_deg is a dead field in this firmware (all zeros), so the
    # turn signal is derived from GPS course (lat/lon gradient), gated to moving
    # frames. Even then it only reaches peak ~0.16, so auto fine-sync genuinely
    # does not lock on this data; the trustworthy alignment is the timestamp
    # anchor (see below) and the xcorr residual is only applied when confident.
    t_abs, accx, speed, steer, gear, lat, lon = [], [], [], [], [], [], []
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
            accx.append(f.accel_x); speed.append(f.speed_mps)
            steer.append(f.steering_deg); gear.append(f.gear)
            lat.append(f.lat); lon.append(f.lon)
    t_abs = np.asarray(t_abs)
    # np.interp (inside resample_uniform) silently returns garbage when its xp
    # (time) array is not strictly increasing. Overlapping Tesla front-clip
    # boundaries can make t_abs non-monotonic, so sort all parallel arrays by
    # t_abs together and drop non-increasing duplicate timestamps.
    _n0 = len(t_abs)
    _was_mono = bool(np.all(np.diff(t_abs) > 0)) if _n0 > 1 else True
    order = np.argsort(t_abs, kind="stable")
    t_abs = t_abs[order]
    accx = np.asarray(accx)[order]; speed = np.asarray(speed)[order]
    steer = np.asarray(steer)[order]; gear = np.asarray(gear)[order]
    lat = np.asarray(lat)[order]; lon = np.asarray(lon)[order]
    keep = np.concatenate(([True], np.diff(t_abs) > 1e-6))
    t_abs = t_abs[keep]; accx = accx[keep]; speed = speed[keep]
    steer = steer[keep]; gear = gear[keep]; lat = lat[keep]; lon = lon[keep]
    print(f"[sync] tesla samples {_n0} -> {len(t_abs)} "
          f"(already monotonic: {_was_mono})", file=sys.stderr)

    # Tesla turn signal from GPS course, gated to moving frames.
    gt, latu = S.resample_uniform(t_abs, lat, rate)
    _, lonu = S.resample_uniform(t_abs, lon, rate)
    _, spdu = S.resample_uniform(t_abs, speed, rate)
    dlat = np.gradient(latu)
    dlon = np.gradient(lonu * np.cos(np.radians(latu)))
    course = np.degrees(np.unwrap(np.arctan2(dlon, dlat)))
    tesla_yawmag = np.abs(np.gradient(course, 1.0 / rate))
    tesla_yawmag[spdu < 2.0] = 0.0          # course is noise when slow/stopped

    # --- Insta360 gyro magnitude on the SAME epoch grid ---
    imu = insta360.read_imu(insv[0])
    it = np.asarray([s.t_s for s in imu]); it = it - it[0]   # epoch ~ insta video start
    gx = np.asarray([s.gyro[0] for s in imu])
    gy = np.asarray([s.gyro[1] for s in imu])
    gz = np.asarray([s.gyro[2] for s in imu])
    # Monotonic guard for the IMU clock (should already be monotonic, guard anyway).
    _imu_mono = bool(np.all(np.diff(it) > 0)) if len(it) > 1 else True
    iorder = np.argsort(it, kind="stable")
    it = it[iorder]; gx = gx[iorder]; gy = gy[iorder]; gz = gz[iorder]
    ikeep = np.concatenate(([True], np.diff(it) > 1e-6))
    it = it[ikeep]; gx = gx[ikeep]; gy = gy[ikeep]; gz = gz[ikeep]
    print(f"[sync] imu samples {len(it)} (already monotonic: {_imu_mono})", file=sys.stderr)
    igt, gxu = S.resample_uniform(it, gx, rate)
    _, gyu = S.resample_uniform(it, gy, rate)
    _, gzu = S.resample_uniform(it, gz, rate)
    gyro_mag = np.sqrt(gxu ** 2 + gyu ** 2 + gzu ** 2)

    # --- timestamp anchor (the trustworthy default) ---
    # tesla_concat local t=0 is fronts[0] start; insta local t=0 is start_jst.
    tesla0 = (S._front_start(fronts[0].name) - start_jst).total_seconds()  # ~ -53 (tesla leads)
    anchor_tesla_lead = -tesla0                                            # ~ +53 seconds

    # --- residual xcorr correction, applied ONLY if confident and small ---
    # Both gt and igt are epoch-relative seconds, so put both magnitude signals
    # on a common overlap grid and correlate with a SMALL max_lag (+/-15 s); a
    # perfect alignment then gives lag~0.
    lo = max(gt[0], igt[0]); hi = min(gt[-1], igt[-1])
    grid = np.arange(lo, hi, 1.0 / rate)
    tv = np.interp(grid, gt, tesla_yawmag)
    iv = np.interp(grid, igt, gyro_mag)
    lag, peak = S.normalized_xcorr(tv, iv, max_lag=int(15 * rate))
    residual = lag / rate
    CONF_MIN = 0.35
    if peak < CONF_MIN or abs(residual) > 15.0:
        residual = 0.0
        method = "anchor"
    else:
        method = "anchor+xcorr"
    tesla_lead = anchor_tesla_lead + residual
    print(f"[sync] anchor_tesla_lead={anchor_tesla_lead:.1f}s residual={residual:.2f}s "
          f"peak={peak:.3f} method={method} -> tesla_lead={tesla_lead:.1f}s", file=sys.stderr)

    res = S.SyncResult(delta_s=tesla_lead, confidence=float(peak),
                       signal="gps_yaw|gyro", anchor_guess=anchor_tesla_lead)

    # --- render window (capped by max_duration) ---
    out_dir = out_root / "sync" / recent
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = encoder
    cap = max_duration if max_duration else min(insv_total, 600.0)

    # Tesla concat: enough clips to cover [0, anchor_tesla_lead + cap], since
    # Tesla leads and we trim it by ~tesla_lead before showing the cap window.
    tesla_cat = out_dir / "tesla-concat.mp4"
    n_clips = min(len(fronts), max(1, int((anchor_tesla_lead + cap) // 60) + 2))
    concat_clips(fronts[:n_clips], tesla_cat, encoder=enc, bitrate_kbps=bitrate_kbps)

    # Insta display: provided flat export, else v360 reframe of the capped window.
    if insta_flat:
        display = Path(insta_flat)
    else:
        display = out_dir / "insta-flat.mp4"
        reframe_insv(insv[0], display, yaw=insta_yaw, pitch=insta_pitch,
                     roll=insta_roll, h_fov=insta_hfov, v_fov=insta_vfov,
                     out_w=1280, out_h=720,
                     encoder=enc, bitrate_kbps=bitrate_kbps,
                     start=0.0, duration=cap)

    # Telemetry rows (~1 Hz over the cap), aligned to the displayed window. The
    # Tesla panel shows footage starting at epoch ~tesla_lead (= anchor_tesla_lead
    # when method=='anchor'), so sample speed/steer/gear over [tesla_lead, +cap]
    # but label rows with local display time 0..cap.
    # Resample speed and steer onto the same gt grid used for GPS/yaw above.
    _, speed_u = S.resample_uniform(t_abs, speed, rate)
    _, steer_u = S.resample_uniform(t_abs, steer, rate)
    # Gear is categorical — map each gt grid point to the nearest real sample.
    gear_idx = np.clip(np.searchsorted(t_abs, gt), 0, len(gear) - 1)
    rows = []
    step = max(1, int(rate))
    for k in range(0, len(gt), step):
        local = gt[k] - tesla_lead
        if local < 0:
            continue
        if local > cap:
            break
        rows.append((float(local), float(speed_u[k]), float(steer_u[k]), str(gear[gear_idx[k]])))
    ass = out_dir / "telemetry.ass"
    ass.write_text(S.telemetry_ass(rows, play_w=2560, play_h=720))

    # Alignment trims. left = insta display, right = tesla concat. Tesla leads by
    # tesla_lead seconds, so trim the tesla (right) input to drop its head; the
    # insta (left) starts at 0. (If tesla_lead were negative, trim insta instead.)
    if tesla_lead >= 0:
        right_start = tesla_lead; left_start = 0.0
    else:
        left_start = -tesla_lead; right_start = 0.0
    print(f"[sync] left_start={left_start:.3f} right_start={right_start:.3f}", file=sys.stderr)

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
        insta_vfov=args.insta_vfov)


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
