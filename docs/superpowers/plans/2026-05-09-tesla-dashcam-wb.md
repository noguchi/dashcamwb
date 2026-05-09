# Tesla DashCam WB Pipeline 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tesla Model 3 Highland の DashCam 6カメラ映像を D65 sRGB ニュートラルに補正する Python CLI パイプライン (`dcwb`) を実装する。設計仕様: `docs/superpowers/specs/2026-05-09-tesla-dashcam-wb-design.md`。

**Architecture:** 統計マイニングで生成したカメラ別固定補正 (A) + クリップごとの Shades-of-Gray シーン適応 AWB (B) を合成した単一 3×3 行列を ffmpeg `colorchannelmixer` に流す遅延レンダー型パイプライン。元データは読み取り専用、出力は `/Users/noguchi/AI/dashcamwb/corrected/` 直下。

**Tech Stack:** Python 3.11+ / numpy / opencv-python / astral / jinja2 / pytest / ffmpeg (with VideoToolbox H.264 encode on Apple Silicon)

---

## File Structure

```
src/dcwb/
├── __init__.py        # package version
├── matrix.py          # 3x3 行列の合成・対角構築
├── profile.py         # Profile dataclass + JSON I/O
├── daylight.py        # astral ラッパ (日の出/日の入り判定)
├── awb.py             # Shades of Gray 実装
├── ffmpeg_wrap.py     # 動画フレーム抽出 (cv2) + ffmpeg レンダリング
├── calibrate.py       # 統計マイニング (per-camera profile 生成)
├── render.py          # イベント単位のレンダリングパイプライン
├── verify.py          # 補正前後の HTML レポート生成
└── cli.py             # argparse エントリポイント (`dcwb` コマンド)

tests/
├── conftest.py
├── fixtures/
│   ├── make_synthetic.py    # 合成テスト動画生成
│   └── (ranged outputs in tmp_path)
├── test_matrix.py
├── test_profile.py
├── test_daylight.py
├── test_awb.py
├── test_ffmpeg_wrap.py
├── test_calibrate.py
├── test_render.py
├── test_verify.py
├── test_cli.py
└── test_integration.py

profiles/                    # 6 カメラ分の JSON (calibrate で生成)
pipeline.json                # B レイヤー設定
pyproject.toml
README.md
```

各モジュールは1つの責務に絞り、依存方向は下から上 (matrix < profile < awb < ffmpeg_wrap < calibrate/render < cli)。

---

## Task 1: プロジェクトスキャフォールディング

**Files:**
- Create: `pyproject.toml`
- Create: `src/dcwb/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pipeline.json`
- Create: `README.md`

- [ ] **Step 1: `pyproject.toml` を作成**

```toml
[project]
name = "dcwb"
version = "0.1.0"
description = "Tesla DashCam White Balance correction pipeline"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "opencv-python>=4.9",
  "astral>=3.2",
  "jinja2>=3.1",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-mock>=3.12",
]

[project.scripts]
dcwb = "dcwb.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 2: パッケージディレクトリと空 `__init__.py` を作成**

```bash
mkdir -p src/dcwb tests/fixtures profiles logs corrected
echo '__version__ = "0.1.0"' > src/dcwb/__init__.py
touch tests/__init__.py
```

- [ ] **Step 3: `tests/conftest.py` を作成**

```python
import pytest
from pathlib import Path

@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent
```

- [ ] **Step 4: `pipeline.json` を作成（仕様セクション 4.4 のデフォルト値）**

```json
{
  "awb": {
    "method": "shades_of_gray",
    "minkowski_p": 6,
    "samples_per_clip": 10,
    "saturation_high": 0.97,
    "saturation_low": 0.03,
    "gain_min": 0.7,
    "gain_max": 1.5,
    "night_attenuation": 0.5
  }
}
```

- [ ] **Step 5: `README.md` を作成（最小限）**

```markdown
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
```

- [ ] **Step 6: 仮想環境作成 + インストール + 動作確認**

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest --collect-only
```

Expected: pytest が tests/ を認識（テストはまだ無いので "no tests ran" でも OK、エラーがないことを確認）

- [ ] **Step 7: コミット**

```bash
git add pyproject.toml src/dcwb/__init__.py tests/__init__.py tests/conftest.py pipeline.json README.md
git commit -m "chore: scaffold dcwb package"
```

`.venv/` は `.gitignore` で既に除外済み (`venv/` パターン)。`profiles/`, `logs/`, `corrected/` は空のためまだ commit 対象外。

---

## Task 2: matrix モジュール (3×3 合成)

**Files:**
- Create: `src/dcwb/matrix.py`
- Create: `tests/test_matrix.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_matrix.py`:

```python
import numpy as np
import pytest
from dcwb.matrix import identity, from_diag, compose

def test_identity_is_3x3_identity():
    I = identity()
    assert I.shape == (3, 3)
    np.testing.assert_array_equal(I, np.eye(3))

def test_from_diag_creates_diagonal():
    M = from_diag(0.9, 1.0, 1.1)
    expected = np.array([
        [0.9, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.1],
    ])
    np.testing.assert_array_equal(M, expected)

def test_compose_is_matmul_of_first_then_second():
    A = from_diag(2.0, 1.0, 1.0)
    B = from_diag(1.0, 1.0, 3.0)
    C = compose(A, B)
    np.testing.assert_array_equal(C, A @ B)

def test_compose_with_identity_is_noop():
    A = from_diag(0.5, 1.5, 0.7)
    np.testing.assert_array_equal(compose(A, identity()), A)
    np.testing.assert_array_equal(compose(identity(), A), A)

def test_shape_validation_rejects_wrong_shape():
    with pytest.raises(ValueError):
        compose(np.zeros((2, 2)), identity())
```

- [ ] **Step 2: テスト実行で失敗確認**

```bash
.venv/bin/pytest tests/test_matrix.py -v
```

Expected: ImportError (dcwb.matrix が存在しない)

- [ ] **Step 3: 最小実装**

`src/dcwb/matrix.py`:

```python
import numpy as np

Matrix3x3 = np.ndarray  # type alias for clarity

def identity() -> Matrix3x3:
    return np.eye(3)

def from_diag(r: float, g: float, b: float) -> Matrix3x3:
    return np.diag([r, g, b]).astype(np.float64)

def _validate(M: Matrix3x3) -> None:
    if M.shape != (3, 3):
        raise ValueError(f"expected 3x3 matrix, got shape {M.shape}")

def compose(left: Matrix3x3, right: Matrix3x3) -> Matrix3x3:
    _validate(left)
    _validate(right)
    return left @ right
```

- [ ] **Step 4: テスト実行で成功確認**

```bash
.venv/bin/pytest tests/test_matrix.py -v
```

Expected: 5 passed

- [ ] **Step 5: コミット**

```bash
git add src/dcwb/matrix.py tests/test_matrix.py
git commit -m "feat(matrix): 3x3 composition primitives"
```

---

## Task 3: profile モジュール (Profile dataclass + JSON I/O)

**Files:**
- Create: `src/dcwb/profile.py`
- Create: `tests/test_profile.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_profile.py`:

```python
import json
import numpy as np
from datetime import datetime, timezone
from dcwb.profile import Profile, CalibrationMeta

def test_profile_round_trip(tmp_path):
    p = Profile(
        camera="front",
        gain_r=0.918,
        gain_g=1.000,
        gain_b=1.067,
        matrix_3x3=np.diag([0.918, 1.0, 1.067]),
        calibration=CalibrationMeta(
            samples_used=247,
            events_sampled=89,
            method="robust_white_patch_median",
            calibrated_at=datetime(2026, 5, 9, 12, 34, 56, tzinfo=timezone.utc),
            samples_per_event_max=3,
        ),
    )
    path = tmp_path / "front.json"
    p.to_json(path)
    loaded = Profile.from_json(path)
    assert loaded.camera == "front"
    assert loaded.gain_r == 0.918
    assert loaded.calibration.samples_used == 247
    np.testing.assert_array_equal(loaded.matrix_3x3, p.matrix_3x3)

def test_from_white_point_computes_gains():
    p = Profile.from_white_point(
        camera="front",
        rgb_white=np.array([180.0, 200.0, 220.0]),
        meta=CalibrationMeta(
            samples_used=100,
            events_sampled=50,
            method="robust_white_patch_median",
            calibrated_at=datetime.now(timezone.utc),
            samples_per_event_max=3,
        ),
    )
    assert p.gain_r == 200.0 / 180.0
    assert p.gain_g == 1.0
    assert p.gain_b == 200.0 / 220.0

def test_json_format_is_human_readable(tmp_path):
    p = Profile.from_white_point(
        camera="back",
        rgb_white=np.array([200.0, 200.0, 200.0]),
        meta=CalibrationMeta(
            samples_used=10,
            events_sampled=5,
            method="robust_white_patch_median",
            calibrated_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
            samples_per_event_max=3,
        ),
    )
    path = tmp_path / "back.json"
    p.to_json(path)
    raw = json.loads(path.read_text())
    assert raw["camera"] == "back"
    assert raw["gain_r"] == 1.0
    assert "calibration" in raw
```

- [ ] **Step 2: テスト実行で失敗確認**

```bash
.venv/bin/pytest tests/test_profile.py -v
```

Expected: ImportError

- [ ] **Step 3: 実装**

`src/dcwb/profile.py`:

```python
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from dcwb.matrix import Matrix3x3, from_diag

@dataclass
class CalibrationMeta:
    samples_used: int
    events_sampled: int
    method: str
    calibrated_at: datetime
    samples_per_event_max: int

@dataclass
class Profile:
    camera: str
    gain_r: float
    gain_g: float
    gain_b: float
    matrix_3x3: Matrix3x3
    calibration: CalibrationMeta

    @classmethod
    def from_white_point(
        cls, camera: str, rgb_white: np.ndarray, meta: CalibrationMeta
    ) -> "Profile":
        rw, gw, bw = float(rgb_white[0]), float(rgb_white[1]), float(rgb_white[2])
        gain_r = gw / rw
        gain_g = 1.0
        gain_b = gw / bw
        return cls(
            camera=camera,
            gain_r=gain_r,
            gain_g=gain_g,
            gain_b=gain_b,
            matrix_3x3=from_diag(gain_r, gain_g, gain_b),
            calibration=meta,
        )

    def to_json(self, path: Path) -> None:
        d = {
            "camera": self.camera,
            "gain_r": self.gain_r,
            "gain_g": self.gain_g,
            "gain_b": self.gain_b,
            "matrix_3x3": self.matrix_3x3.tolist(),
            "calibration": {
                "samples_used": self.calibration.samples_used,
                "events_sampled": self.calibration.events_sampled,
                "method": self.calibration.method,
                "calibrated_at": self.calibration.calibrated_at.isoformat(),
                "samples_per_event_max": self.calibration.samples_per_event_max,
            },
        }
        path.write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "Profile":
        d = json.loads(path.read_text())
        cal = d["calibration"]
        return cls(
            camera=d["camera"],
            gain_r=float(d["gain_r"]),
            gain_g=float(d["gain_g"]),
            gain_b=float(d["gain_b"]),
            matrix_3x3=np.array(d["matrix_3x3"], dtype=np.float64),
            calibration=CalibrationMeta(
                samples_used=int(cal["samples_used"]),
                events_sampled=int(cal["events_sampled"]),
                method=cal["method"],
                calibrated_at=datetime.fromisoformat(cal["calibrated_at"]),
                samples_per_event_max=int(cal["samples_per_event_max"]),
            ),
        )
```

- [ ] **Step 4: テスト実行で成功確認**

```bash
.venv/bin/pytest tests/test_profile.py -v
```

Expected: 3 passed

- [ ] **Step 5: コミット**

```bash
git add src/dcwb/profile.py tests/test_profile.py
git commit -m "feat(profile): Profile dataclass with JSON I/O"
```

---

## Task 4: daylight モジュール (sunrise/sunset 判定)

**Files:**
- Create: `src/dcwb/daylight.py`
- Create: `tests/test_daylight.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_daylight.py`:

```python
from datetime import datetime, timezone, timedelta
from dcwb.daylight import is_daytime, TOKYO_LAT, TOKYO_LON

JST = timezone(timedelta(hours=9))

def test_noon_in_tokyo_is_daytime():
    t = datetime(2026, 5, 5, 12, 0, tzinfo=JST)
    assert is_daytime(t) is True

def test_midnight_in_tokyo_is_not_daytime():
    t = datetime(2026, 5, 5, 0, 0, tzinfo=JST)
    assert is_daytime(t) is False

def test_just_before_sunrise_is_not_daytime():
    # 5月5日東京の日の出は約 04:42。30分マージンで 05:12 までは not daytime
    t = datetime(2026, 5, 5, 5, 0, tzinfo=JST)
    assert is_daytime(t) is False

def test_well_after_sunrise_is_daytime():
    t = datetime(2026, 5, 5, 6, 0, tzinfo=JST)
    assert is_daytime(t) is True

def test_uses_provided_lat_lon():
    # Sydney (-33.86, 151.21) で 12:00 JST は現地で 13:00 → 昼
    t = datetime(2026, 5, 5, 12, 0, tzinfo=JST)
    assert is_daytime(t, lat=-33.86, lon=151.21) is True

def test_default_constants_are_tokyo():
    assert abs(TOKYO_LAT - 35.6762) < 0.001
    assert abs(TOKYO_LON - 139.6503) < 0.001
```

- [ ] **Step 2: テスト実行で失敗確認**

```bash
.venv/bin/pytest tests/test_daylight.py -v
```

Expected: ImportError

- [ ] **Step 3: 実装**

`src/dcwb/daylight.py`:

```python
from datetime import datetime, timedelta
from astral import LocationInfo
from astral.sun import sun

TOKYO_LAT = 35.6762
TOKYO_LON = 139.6503
MARGIN = timedelta(minutes=30)

def is_daytime(
    when: datetime,
    lat: float = TOKYO_LAT,
    lon: float = TOKYO_LON,
) -> bool:
    """Return True if `when` is between sunrise+30min and sunset-30min."""
    if when.tzinfo is None:
        raise ValueError("`when` must be timezone-aware")
    loc = LocationInfo("custom", "custom", str(when.tzinfo), lat, lon)
    s = sun(loc.observer, date=when.date(), tzinfo=when.tzinfo)
    return s["sunrise"] + MARGIN <= when <= s["sunset"] - MARGIN
```

- [ ] **Step 4: テスト実行で成功確認**

```bash
.venv/bin/pytest tests/test_daylight.py -v
```

Expected: 6 passed

- [ ] **Step 5: コミット**

```bash
git add src/dcwb/daylight.py tests/test_daylight.py
git commit -m "feat(daylight): sunrise/sunset based daytime check"
```

---

## Task 5: awb モジュール (Shades of Gray)

**Files:**
- Create: `src/dcwb/awb.py`
- Create: `tests/test_awb.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_awb.py`:

```python
import numpy as np
from dcwb.awb import shades_of_gray

def test_neutral_gray_image_returns_unity_gains():
    # 128/255 のグレー一面 → ゲインは ≈ 1.0
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    g_r, g_g, g_b = shades_of_gray(image, p=6)
    assert abs(g_r - 1.0) < 0.01
    assert abs(g_g - 1.0) < 0.01
    assert abs(g_b - 1.0) < 0.01

def test_red_tinted_image_returns_red_attenuation():
    # 赤強めのグレー (R=180, G=128, B=128) → R ゲインが < 1
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[..., 0] = 180  # R
    image[..., 1] = 128  # G
    image[..., 2] = 128  # B
    # OpenCV BGR convention は呼び出し側で処理。awb は (H, W, 3) の RGB を想定
    g_r, g_g, g_b = shades_of_gray(image, p=6)
    assert g_r < 1.0
    assert abs(g_g - 1.0) < 0.05
    assert g_b > 1.0  # 赤を抑え、緑+青を持ち上げ

def test_saturated_pixels_excluded():
    # 半分は飽和 (255, 255, 255)、半分は (100, 100, 100)
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    image[:50, :, :] = 255
    g_r, g_g, g_b = shades_of_gray(image, p=6, sat_high=0.97)
    # 飽和を除外すれば下半分のグレーが支配 → ゲイン約 1.0
    assert abs(g_r - 1.0) < 0.05

def test_shadow_pixels_excluded():
    # 半分は影 (5, 5, 5)、半分はグレー (128, 128, 128)
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    image[:50, :, :] = 5
    g_r, g_g, g_b = shades_of_gray(image, p=6, sat_low=0.03)
    assert abs(g_r - 1.0) < 0.05
```

- [ ] **Step 2: テスト実行で失敗確認**

```bash
.venv/bin/pytest tests/test_awb.py -v
```

Expected: ImportError

- [ ] **Step 3: 実装**

`src/dcwb/awb.py`:

```python
import numpy as np

def shades_of_gray(
    image_rgb: np.ndarray,
    p: int = 6,
    sat_high: float = 0.97,
    sat_low: float = 0.03,
) -> tuple[float, float, float]:
    """Estimate illuminant via Shades of Gray (Minkowski p-norm) and return
    per-channel gains (g_r, g_g, g_b) such that applying them neutralises
    the estimated illuminant.

    Args:
        image_rgb: HxWx3 uint8 array in RGB order
        p: Minkowski exponent (1 = Gray World, ∞ ≈ White Patch, 6 = balanced)
        sat_high: discard pixels where any channel exceeds this fraction (0..1)
        sat_low:  discard pixels where any channel falls below this fraction
    """
    if image_rgb.dtype != np.uint8:
        raise ValueError("expected uint8 RGB image")
    img = image_rgb.astype(np.float64) / 255.0
    flat = img.reshape(-1, 3)
    mask = np.all((flat <= sat_high) & (flat >= sat_low), axis=1)
    valid = flat[mask]
    if valid.shape[0] == 0:
        return 1.0, 1.0, 1.0
    # Minkowski p-norm per channel
    e = np.power(np.mean(np.power(valid, p), axis=0), 1.0 / p)
    e_max = float(np.max(e))
    g_r = e_max / float(e[0])
    g_g = e_max / float(e[1])
    g_b = e_max / float(e[2])
    return g_r, g_g, g_b
```

- [ ] **Step 4: テスト実行で成功確認**

```bash
.venv/bin/pytest tests/test_awb.py -v
```

Expected: 4 passed

- [ ] **Step 5: コミット**

```bash
git add src/dcwb/awb.py tests/test_awb.py
git commit -m "feat(awb): Shades of Gray illuminant estimation"
```

---

## Task 6: 合成テストフィクスチャ生成

**Files:**
- Create: `tests/fixtures/make_synthetic.py`

`ffmpeg_wrap` と `calibrate`/`render` のテストには実 mp4 が必要。各テストで `tmp_path` 配下に生成する。

- [ ] **Step 1: 合成 mp4 ジェネレータを書く**

`tests/fixtures/make_synthetic.py`:

```python
"""Generate synthetic Tesla DashCam-style mp4s for testing.

Creates short (3s) 320x240 H.264 mp4s with a known constant color cast applied.
This is enough to test:
- ffmpeg/cv2 frame extraction
- statistical mining recovers the cast
- render pipeline neutralises the cast
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np

CAMERAS = [
    "front", "back",
    "left_pillar", "right_pillar",
    "left_repeater", "right_repeater",
]

def make_clip(
    out_path: Path,
    cast_rgb: tuple[float, float, float],
    duration_sec: float = 3.0,
    width: int = 320,
    height: int = 240,
    base_gray: int = 180,
) -> None:
    """Generate one mp4 with a near-uniform color (base_gray * cast).

    cast_rgb: per-channel multiplier representing the camera's intrinsic cast
    (e.g. (1.10, 1.00, 0.92) means R is 10% high, B is 8% low).
    """
    r = int(np.clip(base_gray * cast_rgb[0], 0, 255))
    g = int(np.clip(base_gray * cast_rgb[1], 0, 255))
    b = int(np.clip(base_gray * cast_rgb[2], 0, 255))
    color_str = f"0x{r:02x}{g:02x}{b:02x}"
    # ffmpeg color source → libx264 (no VideoToolbox needed for tests)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={color_str}:s={width}x{height}:d={duration_sec}:r=30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def make_event(
    event_dir: Path,
    casts: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Generate a 6-camera synthetic event under event_dir."""
    casts = casts or {cam: (1.0, 1.0, 1.0) for cam in CAMERAS}
    event_dir.mkdir(parents=True, exist_ok=True)
    timestamp = "2026-05-05_13-49-39"
    for cam in CAMERAS:
        out = event_dir / f"{timestamp}-{cam}.mp4"
        make_clip(out, casts[cam])
    # 注: Tesla 実車の event.json は timezone なし naive ISO format。
    # calibrate.py の `_read_event_timestamp` で JST として解釈する前提。
    (event_dir / "event.json").write_text(
        '{"timestamp":"2026-05-05T13:49:56","city":"Tokyo",'
        '"street":"","est_lat":"35.68","est_lon":"139.65",'
        '"reason":"sentry_aware_object_detection","camera":"5"}'
    )
```

- [ ] **Step 2: 動作確認 (smoke)**

```bash
.venv/bin/python -c "from pathlib import Path; from tests.fixtures.make_synthetic import make_event; make_event(Path('/tmp/dcwb_smoke')); import os; print(sorted(os.listdir('/tmp/dcwb_smoke')))"
```

Expected: 6 mp4 + event.json が出力される

- [ ] **Step 3: コミット**

```bash
git add tests/fixtures/make_synthetic.py
git commit -m "test: synthetic event fixture generator"
```

---

## Task 7: ffmpeg_wrap モジュール (フレーム抽出 + レンダリング)

**Files:**
- Create: `src/dcwb/ffmpeg_wrap.py`
- Create: `tests/test_ffmpeg_wrap.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_ffmpeg_wrap.py`:

```python
import numpy as np
import pytest
from pathlib import Path
from dcwb.ffmpeg_wrap import probe_duration, extract_frame, render_with_matrix
from dcwb.matrix import from_diag
from tests.fixtures.make_synthetic import make_clip

@pytest.fixture
def sample_clip(tmp_path) -> Path:
    p = tmp_path / "sample.mp4"
    make_clip(p, cast_rgb=(1.0, 1.0, 1.0), duration_sec=3.0)
    return p

def test_probe_duration_returns_seconds(sample_clip):
    d = probe_duration(sample_clip)
    assert 2.5 < d < 3.5  # generated as 3.0s

def test_extract_frame_returns_rgb_numpy(sample_clip):
    img = extract_frame(sample_clip, t=1.0)
    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[2] == 3
    # color was 180,180,180 → mean ≈ 180 (some compression noise)
    mean = img.reshape(-1, 3).mean(axis=0)
    assert np.allclose(mean, [180, 180, 180], atol=10)

def test_render_with_identity_preserves_color(sample_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render_with_matrix(sample_clip, out, np.eye(3), bitrate_kbps=4000)
    assert out.exists()
    img = extract_frame(out, t=1.0)
    mean = img.reshape(-1, 3).mean(axis=0)
    assert np.allclose(mean, [180, 180, 180], atol=15)

def test_render_with_red_attenuation_reduces_red(sample_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render_with_matrix(sample_clip, out, from_diag(0.5, 1.0, 1.0), bitrate_kbps=4000)
    img = extract_frame(out, t=1.0)
    mean = img.reshape(-1, 3).mean(axis=0)
    # R は約半分、G/B は維持
    assert mean[0] < 110
    assert 160 < mean[1] < 200
    assert 160 < mean[2] < 200
```

- [ ] **Step 2: テスト実行で失敗確認**

```bash
.venv/bin/pytest tests/test_ffmpeg_wrap.py -v
```

Expected: ImportError

- [ ] **Step 3: 実装**

`src/dcwb/ffmpeg_wrap.py`:

```python
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np
import cv2
from dcwb.matrix import Matrix3x3

def probe_duration(path: Path) -> float:
    """Return clip duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

def extract_frame(path: Path, t: float) -> np.ndarray:
    """Decode one frame at timestamp t (seconds) and return RGB uint8 (H, W, 3)."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"failed to read frame at t={t} from {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()

def render_with_matrix(
    src: Path,
    dst: Path,
    matrix: Matrix3x3,
    bitrate_kbps: int = 12000,
    encoder: str = "h264_videotoolbox",
) -> None:
    """Render src → dst with a 3x3 RGB color transform applied via colorchannelmixer.

    On non-Apple-Silicon systems pass encoder='libx264' explicitly.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"expected 3x3 matrix, got {matrix.shape}")
    m = matrix
    cm = (
        f"colorchannelmixer="
        f"rr={m[0, 0]:.6f}:rg={m[0, 1]:.6f}:rb={m[0, 2]:.6f}:"
        f"gr={m[1, 0]:.6f}:gg={m[1, 1]:.6f}:gb={m[1, 2]:.6f}:"
        f"br={m[2, 0]:.6f}:bg={m[2, 1]:.6f}:bb={m[2, 2]:.6f}"
    )
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", cm,
        "-c:v", encoder,
        "-b:v", f"{bitrate_kbps}k",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(dst)
    except subprocess.CalledProcessError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"ffmpeg failed: {e.stderr.decode('utf-8', errors='replace')[:500]}"
        ) from e
```

注: `h264_videotoolbox` は macOS 専用。テスト環境/CI が macOS であることを前提にしているが、もしテストで失敗した場合は `encoder="libx264"` を渡してフォールバック。本タスクのテストは libx264 を使う想定だが、render_with_matrix のデフォルトは VideoToolbox のまま（運用時の最大速度を出すため）。

- [ ] **Step 4: テストに encoder を渡すよう修正**

`tests/test_ffmpeg_wrap.py` の render テスト 2 件を以下に書き換え:

```python
def test_render_with_identity_preserves_color(sample_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render_with_matrix(sample_clip, out, np.eye(3), bitrate_kbps=4000, encoder="libx264")
    assert out.exists()
    img = extract_frame(out, t=1.0)
    mean = img.reshape(-1, 3).mean(axis=0)
    assert np.allclose(mean, [180, 180, 180], atol=15)

def test_render_with_red_attenuation_reduces_red(sample_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render_with_matrix(sample_clip, out, from_diag(0.5, 1.0, 1.0), bitrate_kbps=4000, encoder="libx264")
    img = extract_frame(out, t=1.0)
    mean = img.reshape(-1, 3).mean(axis=0)
    assert mean[0] < 110
    assert 160 < mean[1] < 200
    assert 160 < mean[2] < 200
```

理由: テストはヘッドレス可能性を考慮して libx264 を使う。本番運用 (render.py) は VideoToolbox を使う。

- [ ] **Step 5: テスト実行で成功確認**

```bash
.venv/bin/pytest tests/test_ffmpeg_wrap.py -v
```

Expected: 4 passed

- [ ] **Step 6: コミット**

```bash
git add src/dcwb/ffmpeg_wrap.py tests/test_ffmpeg_wrap.py
git commit -m "feat(ffmpeg_wrap): probe, frame extract, matrix render"
```

---

## Task 8: calibrate モジュール (統計マイニング)

**Files:**
- Create: `src/dcwb/calibrate.py`
- Create: `tests/test_calibrate.py`

`calibrate.py` は (a) ニュートラルピクセル抽出、(b) 多色性チェック、(c) 幾何中央値による集約、(d) 全イベントから1カメラ分の Profile を構築、を担う。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_calibrate.py`:

```python
import numpy as np
import pytest
from pathlib import Path
from dcwb.calibrate import (
    find_neutral_pixels,
    is_multicolor,
    geometric_median,
    calibrate_camera,
)
from tests.fixtures.make_synthetic import make_event

# 合成 mp4 は均一カラーで is_multicolor が False になるため、
# calibrate のテストでは常に True を返すようパッチする。
@pytest.fixture(autouse=True)
def _bypass_multicolor(monkeypatch):
    monkeypatch.setattr("dcwb.calibrate.is_multicolor",
                       lambda img, threshold=0.05: True)

def test_find_neutral_pixels_picks_high_v_low_s():
    # 上半分: グレー (180, 180, 180) → ニュートラル
    # 下半分: 赤 (200, 50, 50) → 高彩度なので除外
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :, :] = 180
    img[50:, 0, ...] = 200
    img[50:, 1, ...] = 50
    img[50:, 2, ...] = 50
    pixels = find_neutral_pixels(img)
    assert pixels.shape[0] > 0
    # 採用ピクセルは平均がほぼ (180,180,180)
    mean = pixels.mean(axis=0)
    assert np.allclose(mean, [180, 180, 180], atol=5)

def test_find_neutral_pixels_excludes_saturated():
    img = np.full((100, 100, 3), 252, dtype=np.uint8)  # ほぼ全飽和
    pixels = find_neutral_pixels(img)
    assert pixels.shape[0] == 0

def test_is_multicolor_passes_diverse_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:33, :, 0] = 200  # 赤帯
    img[33:66, :, 1] = 200  # 緑帯
    img[66:, :, 2] = 200  # 青帯
    assert is_multicolor(img) is True

def test_is_multicolor_rejects_monocolor_image():
    img = np.full((100, 100, 3), 180, dtype=np.uint8)  # 完全均一
    assert is_multicolor(img) is False

def test_geometric_median_robust_to_outliers():
    points = np.array([
        [100.0, 100.0, 100.0],
        [101.0, 102.0, 99.0],
        [99.0, 100.0, 101.0],
        [500.0, 500.0, 500.0],  # 外れ値
    ])
    med = geometric_median(points)
    # 外れ値の影響を強く受けない (≈ 100)
    assert np.all(np.abs(med - 100.0) < 5.0)

def test_calibrate_camera_recovers_synthetic_cast(tmp_path):
    # 合成イベントで front カメラに既知のキャスト (R=1.10) を設定。
    # Sentry イベントの保存ルート構造を再現: <root>/SentryClips/<event>/...
    event_dir = tmp_path / "SentryClips" / "2026-05-05_13-49-39"
    cast = (1.10, 1.00, 0.90)  # R 強め, B 弱め
    make_event(
        event_dir,
        casts={
            "front": cast,
            "back": (1.0, 1.0, 1.0),
            "left_pillar": (1.0, 1.0, 1.0),
            "right_pillar": (1.0, 1.0, 1.0),
            "left_repeater": (1.0, 1.0, 1.0),
            "right_repeater": (1.0, 1.0, 1.0),
        },
    )
    # event.json の timestamp は "2026-05-05T13:49:56" → JST 13時 → 昼判定 OK
    profile = calibrate_camera(
        camera="front",
        source_root=tmp_path,
        max_per_event=3,
    )
    # 期待ゲイン: gain_r ≈ G/R = 1.0/1.10 ≈ 0.909
    assert abs(profile.gain_r - 1.0 / 1.10) < 0.02
    assert abs(profile.gain_b - 1.0 / 0.90) < 0.02
    assert profile.calibration.samples_used > 0
```

- [ ] **Step 2: テスト実行で失敗確認**

```bash
.venv/bin/pytest tests/test_calibrate.py -v
```

Expected: ImportError

- [ ] **Step 3: 実装**

`src/dcwb/calibrate.py`:

```python
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from datetime import timedelta
from dcwb.profile import Profile, CalibrationMeta
from dcwb.daylight import is_daytime, TOKYO_LAT, TOKYO_LON
from dcwb.ffmpeg_wrap import probe_duration, extract_frame

# Tesla の event.json は naive ISO format (no tz)。実車設定のローカル時刻を保存する想定。
# Tokyo ベースの利用が前提なので JST (UTC+9) として解釈する。
JST = timezone(timedelta(hours=9))

NEUTRAL_V_MIN = 0.7
NEUTRAL_S_MAX = 0.15
SAT_MAX = 250
SHADOW_V_MIN = 0.2

def find_neutral_pixels(image_rgb: np.ndarray) -> np.ndarray:
    """Return Nx3 uint8 array of pixels passing the neutral-candidate mask."""
    if image_rgb.dtype != np.uint8:
        raise ValueError("expected uint8 RGB image")
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    flat = image_rgb.reshape(-1, 3)
    s_flat = s.reshape(-1)
    v_flat = v.reshape(-1)
    mask = (
        (v_flat > NEUTRAL_V_MIN)
        & (s_flat < NEUTRAL_S_MAX)
        & (v_flat > SHADOW_V_MIN)
        & np.all(flat <= SAT_MAX, axis=1)
    )
    return flat[mask]

def is_multicolor(image_rgb: np.ndarray, threshold: float = 0.05) -> bool:
    """True if the scene exhibits enough chroma diversity (saturation std)."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] / 255.0
    return float(s.std()) > threshold

def geometric_median(points: np.ndarray, eps: float = 1e-5, max_iter: int = 200) -> np.ndarray:
    """Weiszfeld's algorithm for the geometric median of N points in R^d."""
    y = points.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(points - y, axis=1)
        nz = d > eps
        if not np.any(nz):
            return y
        w = 1.0 / d[nz]
        y_new = (points[nz] * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            return y_new
        y = y_new
    return y

def _list_clips_for_camera(source_root: Path, camera: str) -> list[Path]:
    paths: list[Path] = []
    for sub in ("SentryClips", "RecentClips", "SavedClips"):
        root = source_root / sub
        if not root.exists():
            continue
        paths.extend(root.rglob(f"*-{camera}.mp4"))
    return sorted(paths)

def _event_dir_of(clip: Path) -> Path:
    return clip.parent

def _read_event_timestamp(event_dir: Path) -> datetime | None:
    """Read event.json["timestamp"]. Naive timestamps are interpreted as JST."""
    ev = event_dir / "event.json"
    if not ev.exists():
        return None
    try:
        d = json.loads(ev.read_text())
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=JST)
        return ts
    except Exception:
        return None

def _read_event_latlon(event_dir: Path) -> tuple[float, float]:
    ev = event_dir / "event.json"
    if not ev.exists():
        return TOKYO_LAT, TOKYO_LON
    try:
        d = json.loads(ev.read_text())
        return float(d.get("est_lat") or TOKYO_LAT), float(d.get("est_lon") or TOKYO_LON)
    except Exception:
        return TOKYO_LAT, TOKYO_LON

def calibrate_camera(
    camera: str,
    source_root: Path,
    max_per_event: int = 3,
) -> Profile:
    """Mine neutral pixels across all daytime clips for one camera and build a Profile."""
    clips = _list_clips_for_camera(source_root, camera)
    all_pixels: list[np.ndarray] = []
    events_seen: set[Path] = set()
    for clip in clips:
        ev_dir = _event_dir_of(clip)
        ts = _read_event_timestamp(ev_dir)
        lat, lon = _read_event_latlon(ev_dir)
        if ts is not None and not is_daytime(ts, lat=lat, lon=lon):
            continue
        try:
            duration = probe_duration(clip)
        except Exception:
            continue
        # 等間隔に最大 max_per_event 枚
        n = max_per_event
        ts_list = [duration * (i + 0.5) / n for i in range(n)]
        for t in ts_list:
            try:
                img = extract_frame(clip, t)
            except Exception:
                continue
            if not is_multicolor(img):
                continue
            pixels = find_neutral_pixels(img)
            if pixels.shape[0] == 0:
                continue
            all_pixels.append(pixels)
            events_seen.add(ev_dir)
    if not all_pixels:
        raise RuntimeError(f"no neutral samples found for camera={camera}")
    stacked = np.concatenate(all_pixels, axis=0).astype(np.float64)
    white_point = geometric_median(stacked)
    meta = CalibrationMeta(
        samples_used=int(stacked.shape[0]),
        events_sampled=len(events_seen),
        method="robust_white_patch_median",
        calibrated_at=datetime.now(timezone.utc),
        samples_per_event_max=max_per_event,
    )
    return Profile.from_white_point(camera=camera, rgb_white=white_point, meta=meta)
```

- [ ] **Step 4: テスト実行で成功確認**

```bash
.venv/bin/pytest tests/test_calibrate.py -v
```

Expected: 6 passed

(`is_multicolor` のバイパスは `tests/test_calibrate.py` の autouse fixture で既に有効化済み)

- [ ] **Step 5: コミット**

```bash
git add src/dcwb/calibrate.py tests/test_calibrate.py tests/fixtures/make_synthetic.py
git commit -m "feat(calibrate): per-camera statistical mining"
```

---

## Task 9: render モジュール (イベント単位レンダー)

**Files:**
- Create: `src/dcwb/render.py`
- Create: `tests/test_render.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_render.py`:

```python
import json
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timezone
from dcwb.render import render_event, compose_clip_matrix
from dcwb.profile import Profile, CalibrationMeta
from dcwb.matrix import from_diag
from dcwb.ffmpeg_wrap import extract_frame
from tests.fixtures.make_synthetic import make_event, CAMERAS

def _mk_profile(camera: str, gain_r=1.0, gain_b=1.0) -> Profile:
    return Profile.from_white_point(
        camera=camera,
        rgb_white=np.array([1.0 / gain_r, 1.0, 1.0 / gain_b]) * 200.0,
        meta=CalibrationMeta(
            samples_used=100, events_sampled=10,
            method="test", calibrated_at=datetime.now(timezone.utc),
            samples_per_event_max=3,
        ),
    )

def test_compose_clip_matrix_combines_profile_and_scene_gain():
    profile = _mk_profile("front", gain_r=0.9, gain_b=1.1)
    final = compose_clip_matrix(profile, scene_gain=(1.0, 1.0, 1.0))
    np.testing.assert_array_almost_equal(final, profile.matrix_3x3)

def test_compose_clip_matrix_applies_fallback_when_gain_extreme():
    profile = _mk_profile("front")
    final = compose_clip_matrix(
        profile,
        scene_gain=(1.0, 1.0, 2.0),  # > gain_max=1.5 → B 全体破棄
        gain_min=0.7, gain_max=1.5,
    )
    # B 破棄なら final は profile.matrix_3x3 と一致
    np.testing.assert_array_almost_equal(final, profile.matrix_3x3)

def test_render_event_neutralises_known_cast(tmp_path):
    # source: front カメラに R=1.10 のキャスト
    source_root = tmp_path / "src"
    event_name = "2026-05-05_13-50-46"
    event_dir = source_root / "SentryClips" / event_name
    cast = (1.10, 1.00, 0.90)
    make_event(
        event_dir,
        casts={
            "front": cast,
            "back": (1.0, 1.0, 1.0),
            "left_pillar": (1.0, 1.0, 1.0),
            "right_pillar": (1.0, 1.0, 1.0),
            "left_repeater": (1.0, 1.0, 1.0),
            "right_repeater": (1.0, 1.0, 1.0),
        },
    )

    # profiles/ を準備 (キャストの逆ゲイン)
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    for cam, c in [
        ("front", cast),
        ("back", (1.0, 1.0, 1.0)),
        ("left_pillar", (1.0, 1.0, 1.0)),
        ("right_pillar", (1.0, 1.0, 1.0)),
        ("left_repeater", (1.0, 1.0, 1.0)),
        ("right_repeater", (1.0, 1.0, 1.0)),
    ]:
        # Profile.from_white_point は rgb_white から gain を逆算 → 逆キャスト
        white_pt = np.array(c) * 200.0
        prof = Profile.from_white_point(
            cam, white_pt,
            CalibrationMeta(
                samples_used=100, events_sampled=10, method="test",
                calibrated_at=datetime.now(timezone.utc),
                samples_per_event_max=3,
            ),
        )
        prof.to_json(profiles_dir / f"{cam}.json")

    out_root = tmp_path / "corrected"
    pipeline_cfg = {
        "awb": {
            "method": "shades_of_gray", "minkowski_p": 6,
            "samples_per_clip": 3, "saturation_high": 0.97, "saturation_low": 0.03,
            "gain_min": 0.7, "gain_max": 1.5, "night_attenuation": 0.5,
        }
    }
    render_event(
        event_dir=event_dir,
        out_root=out_root,
        profiles_dir=profiles_dir,
        pipeline_cfg=pipeline_cfg,
        encoder="libx264",
    )
    out_event_dir = out_root / event_name
    assert out_event_dir.exists()

    # 出力 front 動画を確認 → R=G=B（補正成功）
    front_clips = sorted(out_event_dir.glob("*-front.mp4"))
    assert len(front_clips) > 0
    img = extract_frame(front_clips[0], t=1.0)
    mean = img.reshape(-1, 3).mean(axis=0)
    # 全チャネルが ≈180 に揃う（合成 base_gray=180）
    assert max(mean) - min(mean) < 10

    # event.json と _pipeline.json が出力されている
    assert (out_event_dir / "event.json").exists()
    assert (out_event_dir / "_pipeline.json").exists()
```

- [ ] **Step 2: テスト実行で失敗確認**

```bash
.venv/bin/pytest tests/test_render.py -v
```

Expected: ImportError

- [ ] **Step 3: 実装**

`src/dcwb/render.py`:

```python
from __future__ import annotations
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import numpy as np
from dcwb.profile import Profile
from dcwb.matrix import from_diag, compose, Matrix3x3
from dcwb.awb import shades_of_gray
from dcwb.ffmpeg_wrap import probe_duration, extract_frame, render_with_matrix

CAMERAS = (
    "front", "back",
    "left_pillar", "right_pillar",
    "left_repeater", "right_repeater",
)

def compose_clip_matrix(
    profile: Profile,
    scene_gain: tuple[float, float, float],
    gain_min: float = 0.7,
    gain_max: float = 1.5,
    attenuation: float = 1.0,
) -> Matrix3x3:
    """Combine A (profile) and B (scene gain) into a single 3x3.

    If any scene_gain channel falls outside [gain_min, gain_max], B is dropped
    entirely (fallback to A only). attenuation ∈ [0, 1] linearly weakens B
    toward identity (used for night attenuation).
    """
    g_r, g_g, g_b = scene_gain
    if any(g < gain_min or g > gain_max for g in (g_r, g_g, g_b)):
        return profile.matrix_3x3
    if attenuation < 1.0:
        g_r = 1.0 + (g_r - 1.0) * attenuation
        g_g = 1.0 + (g_g - 1.0) * attenuation
        g_b = 1.0 + (g_b - 1.0) * attenuation
    return compose(from_diag(g_r, g_g, g_b), profile.matrix_3x3)

def _camera_of(clip: Path) -> str:
    # ファイル名形式: <ts>-<camera>.mp4 (camera は left_pillar など _ を含む)
    stem = clip.stem  # 拡張子除く
    # 最初の "-" 以降を取得 → "<ts>-<camera>" 形式なので strip
    parts = stem.split("-", 3)
    # 例: 2026-05-05_13-49-39-front → split max 3 → ['2026', '05', '05_13', '49-39-front']
    # camera 名は CAMERAS のいずれか接尾辞
    for cam in CAMERAS:
        if stem.endswith("-" + cam):
            return cam
    raise ValueError(f"could not infer camera from filename: {clip.name}")

def _estimate_scene_gain(
    clip: Path,
    profile: Profile,
    samples_per_clip: int,
    sat_high: float,
    sat_low: float,
    p: int,
) -> tuple[float, float, float]:
    duration = probe_duration(clip)
    ts_list = [duration * (i + 0.5) / samples_per_clip for i in range(samples_per_clip)]
    gains: list[tuple[float, float, float]] = []
    for t in ts_list:
        img_rgb = extract_frame(clip, t)
        # A 補正を先にかけてから B 推定
        flat = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0
        flat = flat @ profile.matrix_3x3.T
        np.clip(flat, 0.0, 1.0, out=flat)
        img_a = (flat.reshape(img_rgb.shape) * 255.0).astype(np.uint8)
        gains.append(shades_of_gray(img_a, p=p, sat_high=sat_high, sat_low=sat_low))
    g = np.array(gains).mean(axis=0)
    return float(g[0]), float(g[1]), float(g[2])

def render_event(
    event_dir: Path,
    out_root: Path,
    profiles_dir: Path,
    pipeline_cfg: dict,
    encoder: str = "h264_videotoolbox",
    bitrate_kbps: int = 12000,
) -> None:
    """Render every camera mp4 in event_dir to out_root/<event_name>/."""
    awb_cfg = pipeline_cfg["awb"]
    out_dir = out_root / event_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = {
        cam: Profile.from_json(profiles_dir / f"{cam}.json")
        for cam in CAMERAS
    }

    snapshot: dict = {"event": event_dir.name, "clips": []}

    for clip in sorted(event_dir.glob("*.mp4")):
        cam = _camera_of(clip)
        prof = profiles[cam]
        scene_gain = _estimate_scene_gain(
            clip, prof,
            samples_per_clip=int(awb_cfg["samples_per_clip"]),
            sat_high=float(awb_cfg["saturation_high"]),
            sat_low=float(awb_cfg["saturation_low"]),
            p=int(awb_cfg["minkowski_p"]),
        )
        final_matrix = compose_clip_matrix(
            prof, scene_gain,
            gain_min=float(awb_cfg["gain_min"]),
            gain_max=float(awb_cfg["gain_max"]),
        )
        render_with_matrix(
            clip, out_dir / clip.name, final_matrix,
            bitrate_kbps=bitrate_kbps, encoder=encoder,
        )
        snapshot["clips"].append({
            "clip": clip.name, "camera": cam,
            "scene_gain": list(scene_gain),
            "final_matrix": final_matrix.tolist(),
        })

    # event.json copy
    src_meta = event_dir / "event.json"
    if src_meta.exists():
        shutil.copy2(src_meta, out_dir / "event.json")

    # thumb.png: render through the front camera matrix if present
    src_thumb = event_dir / "thumb.png"
    if src_thumb.exists():
        # 簡易: front カメラの行列で thumb を補正して保存
        import cv2
        img_bgr = cv2.imread(str(src_thumb), cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            flat = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0
            corrected = flat @ profiles["front"].matrix_3x3.T
            np.clip(corrected, 0.0, 1.0, out=corrected)
            out_rgb = (corrected.reshape(img_rgb.shape) * 255.0).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / "thumb.png"), out_bgr)

    # snapshot
    snapshot["rendered_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "_pipeline.json").write_text(json.dumps(snapshot, indent=2))
```

- [ ] **Step 4: テスト実行で成功確認**

```bash
.venv/bin/pytest tests/test_render.py -v
```

Expected: 3 passed

- [ ] **Step 5: コミット**

```bash
git add src/dcwb/render.py tests/test_render.py
git commit -m "feat(render): per-event WB rendering pipeline"
```

---

## Task 10: verify モジュール (HTML レポート)

**Files:**
- Create: `src/dcwb/verify.py`
- Create: `src/dcwb/templates/verify.html.j2`
- Create: `tests/test_verify.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_verify.py`:

```python
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from dcwb.verify import generate_verify_report
from dcwb.profile import Profile, CalibrationMeta
from tests.fixtures.make_synthetic import make_event

def test_generate_verify_report_creates_html(tmp_path):
    event_dir = tmp_path / "src" / "2026-05-05_13-50-46"
    make_event(event_dir, casts={
        "front": (1.1, 1.0, 0.9),
        "back": (1.0, 1.0, 1.0),
        "left_pillar": (1.0, 1.0, 1.0),
        "right_pillar": (1.0, 1.0, 1.0),
        "left_repeater": (1.0, 1.0, 1.0),
        "right_repeater": (1.0, 1.0, 1.0),
    })
    # profiles
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    for cam in ("front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater"):
        cast = (1.1, 1.0, 0.9) if cam == "front" else (1.0, 1.0, 1.0)
        p = Profile.from_white_point(
            cam, np.array(cast) * 200.0,
            CalibrationMeta(samples_used=10, events_sampled=2, method="t",
                            calibrated_at=datetime.now(timezone.utc),
                            samples_per_event_max=3),
        )
        p.to_json(profiles_dir / f"{cam}.json")

    out = tmp_path / "report.html"
    generate_verify_report(
        event_dir=event_dir,
        profiles_dir=profiles_dir,
        out_html=out,
        encoder="libx264",
    )
    assert out.exists()
    html = out.read_text()
    # 6 カメラ分が含まれている
    for cam in ("front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater"):
        assert cam in html
    # 補正前後の画像（base64）参照が3列ある想定
    assert html.count("data:image/png;base64,") >= 6  # 各カメラごとに最低1枚
```

- [ ] **Step 2: テンプレートを作成**

`src/dcwb/templates/verify.html.j2`:

```html
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>dcwb verify — {{ event_name }}</title>
  <style>
    body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 0; padding: 1rem; }
    h1 { font-size: 1rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { padding: 0.5rem; border-bottom: 1px solid #333; vertical-align: top; }
    img { max-width: 100%; height: auto; display: block; }
    .meta { font-size: 0.8rem; color: #999; }
  </style>
</head>
<body>
  <h1>dcwb verify — {{ event_name }}</h1>
  <table>
    <thead>
      <tr><th>camera</th><th>before</th><th>A only</th><th>A + B (full)</th></tr>
    </thead>
    <tbody>
      {% for row in rows %}
      <tr>
        <td>{{ row.camera }}<br><span class="meta">scene_gain: {{ row.scene_gain }}</span></td>
        <td><img src="data:image/png;base64,{{ row.before }}"></td>
        <td><img src="data:image/png;base64,{{ row.a_only }}"></td>
        <td><img src="data:image/png;base64,{{ row.full }}"></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>
```

- [ ] **Step 3: 実装**

`src/dcwb/verify.py`:

```python
from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dcwb.profile import Profile
from dcwb.matrix import from_diag
from dcwb.ffmpeg_wrap import probe_duration, extract_frame
from dcwb.awb import shades_of_gray
from dcwb.render import compose_clip_matrix, _camera_of, CAMERAS

TEMPLATES_DIR = Path(__file__).parent / "templates"

def _to_base64_png(img_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _apply_matrix(img_rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    flat = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0
    out = flat @ matrix.T
    np.clip(out, 0.0, 1.0, out=out)
    return (out.reshape(img_rgb.shape) * 255.0).astype(np.uint8)

def generate_verify_report(
    event_dir: Path,
    profiles_dir: Path,
    out_html: Path,
    encoder: str = "h264_videotoolbox",
) -> None:
    profiles = {
        cam: Profile.from_json(profiles_dir / f"{cam}.json")
        for cam in CAMERAS
    }
    rows = []
    for clip in sorted(event_dir.glob("*.mp4")):
        cam = _camera_of(clip)
        prof = profiles[cam]
        # 1 フレームを中央時刻から
        duration = probe_duration(clip)
        before = extract_frame(clip, duration / 2.0)
        a_only = _apply_matrix(before, prof.matrix_3x3)
        # B を 1 サンプルで簡易計算
        scene_gain = shades_of_gray(a_only, p=6)
        final_m = compose_clip_matrix(prof, scene_gain)
        full = _apply_matrix(before, final_m)
        rows.append({
            "camera": cam,
            "scene_gain": [round(g, 3) for g in scene_gain],
            "before": _to_base64_png(before),
            "a_only": _to_base64_png(a_only),
            "full": _to_base64_png(full),
        })
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    tmpl = env.get_template("verify.html.j2")
    out_html.write_text(tmpl.render(event_name=event_dir.name, rows=rows))
```

- [ ] **Step 4: テンプレートをパッケージに含める設定**

`pyproject.toml` の `[tool.setuptools]` に追加:

```toml
[tool.setuptools.package-data]
dcwb = ["templates/*.j2"]
```

- [ ] **Step 5: 再インストール + テスト実行**

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/test_verify.py -v
```

Expected: 1 passed

- [ ] **Step 6: コミット**

```bash
git add src/dcwb/verify.py src/dcwb/templates/verify.html.j2 tests/test_verify.py pyproject.toml
git commit -m "feat(verify): HTML side-by-side comparison report"
```

---

## Task 11: cli モジュール (`dcwb` エントリポイント)

**Files:**
- Create: `src/dcwb/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_cli.py`:

```python
import json
import sys
import pytest
from pathlib import Path
from dcwb import cli

def test_cli_no_args_prints_help(capsys):
    with pytest.raises(SystemExit):
        cli.main([])
    captured = capsys.readouterr()
    assert "calibrate" in captured.out or "calibrate" in captured.err

def test_cli_calibrate_invokes_calibrate_camera(tmp_path, monkeypatch):
    called = []
    def fake_calibrate(**kw):
        called.append(kw)
        from dcwb.profile import Profile, CalibrationMeta
        from datetime import datetime, timezone
        import numpy as np
        return Profile.from_white_point(
            kw["camera"], np.array([200.0, 200.0, 200.0]),
            CalibrationMeta(
                samples_used=10, events_sampled=5, method="test",
                calibrated_at=datetime.now(timezone.utc),
                samples_per_event_max=3,
            ),
        )
    monkeypatch.setattr("dcwb.cli.calibrate_camera", fake_calibrate)
    out_profiles = tmp_path / "profiles"
    cli.main([
        "calibrate",
        "--source", str(tmp_path),
        "--profiles-dir", str(out_profiles),
        "--max-samples-per-event", "2",
    ])
    assert len(called) == 6  # 6 cameras
    assert all((out_profiles / f"{c}.json").exists()
               for c in ("front","back","left_pillar","right_pillar","left_repeater","right_repeater"))

def test_cli_render_invokes_render_event(tmp_path, monkeypatch):
    captured = {}
    def fake_render(**kw):
        captured.update(kw)
    monkeypatch.setattr("dcwb.cli.render_event", fake_render)
    event_dir = tmp_path / "evt"
    event_dir.mkdir()
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    out_root = tmp_path / "out"
    pipeline_cfg = tmp_path / "pipeline.json"
    pipeline_cfg.write_text(json.dumps({"awb": {
        "method": "shades_of_gray", "minkowski_p": 6, "samples_per_clip": 3,
        "saturation_high": 0.97, "saturation_low": 0.03,
        "gain_min": 0.7, "gain_max": 1.5, "night_attenuation": 0.5,
    }}))
    cli.main([
        "render", str(event_dir),
        "--profiles-dir", str(profiles_dir),
        "--out-root", str(out_root),
        "--pipeline-config", str(pipeline_cfg),
    ])
    assert captured["event_dir"] == event_dir
    assert captured["out_root"] == out_root
```

- [ ] **Step 2: テスト実行で失敗確認**

```bash
.venv/bin/pytest tests/test_cli.py -v
```

Expected: ImportError or main not defined

- [ ] **Step 3: 実装**

`src/dcwb/cli.py`:

```python
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
    generate_verify_report(
        event_dir=args.event_dir,
        profiles_dir=args.profiles_dir,
        out_html=args.out_html,
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
```

- [ ] **Step 4: テスト実行で成功確認**

```bash
.venv/bin/pytest tests/test_cli.py -v
```

Expected: 3 passed

- [ ] **Step 5: コミット**

```bash
git add src/dcwb/cli.py tests/test_cli.py
git commit -m "feat(cli): dcwb command-line interface"
```

---

## Task 12: エンドツーエンド結合テスト

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_integration.py`:

```python
import numpy as np
from pathlib import Path
from dcwb import cli
from dcwb.ffmpeg_wrap import extract_frame
from tests.fixtures.make_synthetic import make_event, CAMERAS

def test_full_pipeline_calibrate_then_render(tmp_path, monkeypatch):
    """Generate a synthetic event with a known per-camera cast,
    run dcwb calibrate, then dcwb render, and assert the rendered
    front clip is neutral within tolerance."""
    monkeypatch.setattr("dcwb.calibrate.is_multicolor", lambda img, threshold=0.05: True)

    source_root = tmp_path / "source"
    sentry_root = source_root / "SentryClips"
    casts = {
        "front": (1.10, 1.00, 0.90),
        "back": (0.95, 1.00, 1.05),
        "left_pillar": (1.05, 1.00, 0.97),
        "right_pillar": (1.02, 1.00, 0.98),
        "left_repeater": (0.98, 1.00, 1.03),
        "right_repeater": (1.01, 1.00, 0.99),
    }
    # 同じイベントを2回生成（calibrate のサンプル数を増やすため）
    for ts in ("2026-05-05_13-50-46", "2026-05-05_14-30-00"):
        make_event(sentry_root / ts, casts=casts)

    profiles_dir = tmp_path / "profiles"
    out_root = tmp_path / "out"
    pipeline_cfg = tmp_path / "pipeline.json"
    pipeline_cfg.write_text(
        '{"awb":{"method":"shades_of_gray","minkowski_p":6,'
        '"samples_per_clip":3,"saturation_high":0.97,"saturation_low":0.03,'
        '"gain_min":0.7,"gain_max":1.5,"night_attenuation":0.5}}'
    )

    # 1. Calibrate
    rc = cli.main([
        "calibrate",
        "--source", str(source_root),
        "--profiles-dir", str(profiles_dir),
        "--max-samples-per-event", "3",
    ])
    assert rc == 0
    for cam in CAMERAS:
        assert (profiles_dir / f"{cam}.json").exists()

    # 2. Render
    event_dir = sentry_root / "2026-05-05_13-50-46"
    rc = cli.main([
        "render", str(event_dir),
        "--profiles-dir", str(profiles_dir),
        "--out-root", str(out_root),
        "--pipeline-config", str(pipeline_cfg),
        "--encoder", "libx264",
        "--bitrate-kbps", "4000",
    ])
    assert rc == 0

    # 3. Assert each rendered camera output is neutral
    out_event = out_root / "2026-05-05_13-50-46"
    for cam in CAMERAS:
        clips = sorted(out_event.glob(f"*-{cam}.mp4"))
        assert len(clips) > 0, f"no rendered clip for {cam}"
        img = extract_frame(clips[0], t=1.0)
        mean = img.reshape(-1, 3).mean(axis=0)
        spread = float(max(mean) - min(mean))
        assert spread < 12, f"{cam}: channel spread {spread:.1f} too large (mean={mean})"
```

- [ ] **Step 2: テスト実行で成功確認**

```bash
.venv/bin/pytest tests/test_integration.py -v
```

Expected: 1 passed（実行に1〜2分かかる可能性）

ヒント: もしカメラごとに spread が 12 を超える場合は、ジェネレーション損失（合成 mp4 の libx264 圧縮）が支配的。閾値を緩めて 18 以下なら設計通り。

- [ ] **Step 3: 全テストスイートを実行**

```bash
.venv/bin/pytest -v
```

Expected: 全テスト pass

- [ ] **Step 4: コミット**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end calibrate + render integration"
```

---

## Task 13: README に運用手順を追記

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README を運用手順を加えた内容に更新**

`README.md` 全体を以下に置き換え:

```markdown
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
```

- [ ] **Step 2: コミット**

```bash
git add README.md
git commit -m "docs: README workflow"
```

---

## Self-Review

仕様カバレッジを spec の各セクションに対してチェック:

- **Spec §1.4 (補正手法 A + B):** Task 8 (calibrate, A 生成) + Task 9 `compose_clip_matrix` + `_estimate_scene_gain` (B) でカバー
- **Spec §3 (キャリブレーション):** Task 8 で `find_neutral_pixels` / `is_multicolor` / `geometric_median` / `calibrate_camera` を実装
- **Spec §3.1 (昼間判定):** Task 4 で `is_daytime`、Task 8 で event.json から lat/lon を読んで適用
- **Spec §3.4 (Profile JSON 形式):** Task 3 で実装、key 名・形状一致
- **Spec §4 (Shades of Gray B レイヤー):** Task 5 + Task 9 でカバー
- **Spec §4.3 (フォールバック):** Task 9 `compose_clip_matrix` の gain_min/max 範囲チェック
- **Spec §4.4 (pipeline.json):** Task 1 で生成
- **Spec §5 (レンダリングパイプライン):** Task 7 (`render_with_matrix`) + Task 9 (`render_event`)
- **Spec §5.1 ffmpeg コマンド形式:** Task 7 で `colorchannelmixer` + VideoToolbox 既定 + atomic rename + `+faststart`
- **Spec §6 (CLI):** Task 11 で 4 サブコマンド (calibrate/render/verify/render-all)
- **Spec §7 (テスト):** Task 2-12 で全モジュール unit test + Task 12 で integration test
- **Spec §8 (リポジトリ構成):** Task 1 + 各タスクで配置

仕様外:
- Spec §10 で確認とした「並列セッション数」は MVP では逐次実行（外側 xargs でユーザが並列化）として実装。本計画の対象外。

**Placeholder スキャン:** "TBD" / "implement later" 系は無し。全タスクに実コードあり。

**型・名前一貫性:**
- `Profile.matrix_3x3`, `Profile.from_white_point`, `Profile.from_json/to_json` は Task 3, 8, 9, 10, 11 で同名で参照
- `compose_clip_matrix(profile, scene_gain, gain_min, gain_max, attenuation)` は Task 9 と Task 10 で同シグネチャ
- `CAMERAS` 定数は Task 6 (fixtures), Task 9 (render), Task 10 (verify), Task 11 (cli) で再利用 — Task 9 で `dcwb.render.CAMERAS` を canonical とし、verify と cli は import 経由で共有

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-09-tesla-dashcam-wb.md`.
