# dcwb — Tesla DashCam White Balance

CLI pipeline that white-balances Tesla Model 3 Highland DashCam footage to D65 sRGB neutral.

Design spec: [`docs/superpowers/specs/2026-05-09-tesla-dashcam-wb-design.md`](docs/superpowers/specs/2026-05-09-tesla-dashcam-wb-design.md)

## Setup

Requires Python 3.11+ and `ffmpeg` (with VideoToolbox on Apple Silicon for runtime; `libx264` for tests).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Workflow

1. **Calibrate** (one-time, re-run if Tesla firmware changes camera color):

```bash
.venv/bin/dcwb calibrate \
  --source /Volumes/sentryusb \
  --profiles-dir profiles \
  --max-samples-per-event 3
```

This mines neutral pixels from all daytime Sentry/Recent/Saved clips and writes `profiles/<camera>.json` for each of the 6 cameras.

2. **Verify** the calibration on a sample event:

```bash
.venv/bin/dcwb verify /Volumes/sentryusb/SentryClips/2026-05-05_13-50-46 \
  --out-html verify.html
open verify.html
```

The HTML shows three columns per camera: original, A-only, A+B (full pipeline). Confirm white objects look neutral and cameras match each other.

3. **Render** an event when you need a corrected output:

```bash
.venv/bin/dcwb render /Volumes/sentryusb/SentryClips/2026-05-05_13-50-46
# → /Users/noguchi/AI/dashcamwb/corrected/2026-05-05_13-50-46/
```

Or batch-render every event in a source directory:

```bash
.venv/bin/dcwb render-all --source /Volumes/sentryusb/SentryClips
```

## Testing

```bash
.venv/bin/pytest -v
```

Tests use synthetic mp4s generated with `libx264` for portability.
