# dcwb — Tesla DashCam White Balance

CLI pipeline that white-balances Tesla Model 3 Highland DashCam footage to D65 sRGB neutral.

## Setup

Requires Python 3.11+ and `ffmpeg` (with VideoToolbox on Apple Silicon).

```bash
python -m venv .venv
source .venv/bin/activate.fish  # fish shell
pip install -e ".[dev]"
```

## Usage

```bash
dcwb calibrate --source /Volumes/sentryusb
dcwb render /Volumes/sentryusb/SentryClips/2026-05-05_13-50-46
dcwb verify /Volumes/sentryusb/SentryClips/2026-05-05_13-50-46
```

See `docs/superpowers/specs/2026-05-09-tesla-dashcam-wb-design.md` for design.
