# VLM-driven drive highlight selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `dcwb highlight-day` select daily front-camera highlight excerpts by an open-weight VLM's per-clip interest score (with scene tags + caption recorded in the manifest), reached over an OpenAI-compatible HTTP API on a LAN GPU host, while keeping the existing MVP scorer as an explicit fallback.

**Architecture:** A new `src/dcwb/vlm.py` isolates the HTTP/VLM I/O boundary (config, frame encoding, structured-output parsing, retry, health check). `src/dcwb/highlight.py` gains an AI scoring path (frame sampling → cached VLM calls → rank by interest) that reuses the existing `plan_excerpts`/`cut_clip`/`concat_clips`, plus an AI manifest writer. `src/dcwb/cli.py` builds the VLM config from `pipeline.json`, runs a health check, and decides between AI, `--allow-no-ai` fallback, or hard error. The MVP scorer is untouched.

**Tech Stack:** Python 3.11+, uv, stdlib `urllib.request` (no new dependency), OpenCV (frame decode/JPEG), numpy, pytest + pytest-mock. ffmpeg with libx264 for tests.

---

## File Structure

- **Create `src/dcwb/vlm.py`** — `VlmConfig`, `ClipDescription`, `VlmUnavailableError`, `VlmClient` (HTTP chat + health check + parse + retry), `encode_frame`, prompt/schema constants, `PROMPT_VERSION`. The only place that talks to the network.
- **Modify `src/dcwb/highlight.py`** — add `AiScore`, frame sampling, cache helpers, `describe_candidates`, AI manifest writer, and an AI branch + `selection` param in `highlight_day`. MVP functions unchanged in behavior.
- **Modify `src/dcwb/cli.py`** — new `highlight-day` flags, config load, health-check/fallback wiring.
- **Modify `pipeline.json`** — add `highlight_ai` section.
- **Create `tests/test_vlm.py`** — unit tests for config merge, frame encoding, parsing, retry, transport-error → `VlmUnavailableError`.
- **Modify `tests/test_highlight.py`** — AI scoring/cache/skip tests + AI `highlight_day` test using an injected fake client.
- **Modify `tests/test_cli.py`** — health-check fallback / hard-error wiring tests.
- **Modify `CLAUDE.md`** — document the AI path, new flags, and `highlight_ai` config.

Throughout: VLM output JSON is `{interest:int 0-10, scene_tags:[str], caption:str, drive_quality:"flowing"|"stop_and_go"|"stopped"}`. The interest→score mapping is `clamp01(interest/10 - (0.3 if drive_quality=="stopped" else 0))`. Cache key is `"<clip_name>:<mtime_ns>"`, valid only when stored `model` and `prompt_version` match.

---

## Task 1: `vlm.py` config, constants, and frame encoding

**Files:**
- Create: `src/dcwb/vlm.py`
- Test: `tests/test_vlm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vlm.py
from __future__ import annotations

import base64

import cv2
import numpy as np


def test_vlm_config_from_dict_merges_defaults_and_ignores_unknown():
    from dcwb.vlm import VlmConfig

    cfg = VlmConfig.from_dict({"model": "my-vlm", "frames_per_clip": 5, "bogus": 1})

    assert cfg.model == "my-vlm"
    assert cfg.frames_per_clip == 5
    # untouched defaults
    assert cfg.endpoint == "http://localhost:1234/v1"
    assert cfg.frame_max_edge == 512
    assert cfg.interest_min == 1
    assert cfg.use_json_schema is True
    assert not hasattr(cfg, "bogus")


def test_vlm_config_from_none_is_all_defaults():
    from dcwb.vlm import VlmConfig

    cfg = VlmConfig.from_dict(None)

    assert cfg.model == "qwen2.5-vl-7b-instruct"


def test_effective_system_prompt_uses_builtin_when_none():
    from dcwb.vlm import DEFAULT_SYSTEM_PROMPT, VlmConfig

    assert VlmConfig.from_dict(None).effective_system_prompt == DEFAULT_SYSTEM_PROMPT
    assert VlmConfig.from_dict({"system_prompt": "x"}).effective_system_prompt == "x"


def test_encode_frame_returns_jpeg_data_uri_within_max_edge():
    from dcwb.vlm import encode_frame

    frame = np.zeros((1000, 2000, 3), dtype=np.uint8)
    frame[:, :, 0] = 200  # red-ish in RGB

    uri = encode_frame(frame, max_edge=512)

    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 512
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_vlm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dcwb.vlm'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/dcwb/vlm.py
from __future__ import annotations

import base64
import dataclasses
from dataclasses import dataclass

import cv2
import numpy as np

PROMPT_VERSION = "1"

DEFAULT_SYSTEM_PROMPT = (
    "あなたはドライブ記録のハイライト編集者です。与えられた前方カメラの数フレームを見て、"
    "後から見返して楽しいドライブ映像かどうかを評価します。流れる風景、特徴的な景観、"
    "変化のある街並み、トンネルや海沿いなどを高く評価し、単調・停止・真っ暗な映像は低く評価します。"
    "指定のJSON形式だけを返してください。"
)

# Returned by the VLM. response_format json_schema mirrors this.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "interest": {"type": "integer", "minimum": 0, "maximum": 10},
        "scene_tags": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "caption": {"type": "string"},
        "drive_quality": {
            "type": "string",
            "enum": ["flowing", "stop_and_go", "stopped"],
        },
    },
    "required": ["interest", "scene_tags", "caption", "drive_quality"],
}

_ALLOWED_DRIVE_QUALITY = {"flowing", "stop_and_go", "stopped"}


class VlmUnavailableError(RuntimeError):
    """The VLM endpoint could not be reached or did not respond usably."""


@dataclass(frozen=True)
class ClipDescription:
    interest: int | None
    scene_tags: list
    caption: str
    drive_quality: str
    parse_failed: bool = False


@dataclass(frozen=True)
class VlmConfig:
    endpoint: str = "http://localhost:1234/v1"
    model: str = "qwen2.5-vl-7b-instruct"
    api_key: str = "lm-studio"
    frames_per_clip: int = 3
    frame_max_edge: int = 512
    timeout_sec: float = 120.0
    max_retries: int = 1
    interest_min: int = 1
    system_prompt: str | None = None
    use_json_schema: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "VlmConfig":
        data = data or {}
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def effective_system_prompt(self) -> str:
        return self.system_prompt or DEFAULT_SYSTEM_PROMPT


def encode_frame(frame_rgb: np.ndarray, max_edge: int) -> str:
    """Resize an RGB frame to fit max_edge, JPEG-encode it, return a data URI."""
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, max_edge / float(max(h, w))) if max(h, w) > 0 else 1.0
    if scale < 1.0:
        frame_rgb = cv2.resize(
            frame_rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("failed to JPEG-encode frame")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_vlm.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/vlm.py tests/test_vlm.py
git commit -m "feat(vlm): add VlmConfig, ClipDescription, and frame encoding"
```

---

## Task 2: `vlm.py` VlmClient — payload, parse, retry, health check

**Files:**
- Modify: `src/dcwb/vlm.py`
- Test: `tests/test_vlm.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_vlm.py
import pytest


def _client(monkeypatch_chat=None, monkeypatch_ping=None, **cfg_kwargs):
    from dcwb.vlm import VlmClient, VlmConfig
    cfg = VlmConfig.from_dict(cfg_kwargs)
    return VlmClient(cfg, chat=monkeypatch_chat, ping=monkeypatch_ping)


def test_describe_parses_valid_json_content():
    content = (
        '{"interest": 8, "scene_tags": ["海沿い", "夕焼け"], '
        '"caption": "夕暮れの海岸を流す", "drive_quality": "flowing"}'
    )
    client = _client(monkeypatch_chat=lambda payload: content)

    desc = client.describe(["data:image/jpeg;base64,AAAA"])

    assert desc.parse_failed is False
    assert desc.interest == 8
    assert desc.scene_tags == ["海沿い", "夕焼け"]
    assert desc.caption == "夕暮れの海岸を流す"
    assert desc.drive_quality == "flowing"


def test_describe_strips_code_fence_and_extra_text():
    content = '```json\n{"interest": 3, "scene_tags": [], "caption": "x", "drive_quality": "stopped"}\n```'
    client = _client(monkeypatch_chat=lambda payload: content)

    desc = client.describe(["uri"])

    assert desc.interest == 3
    assert desc.drive_quality == "stopped"


def test_describe_retries_then_marks_parse_failed():
    calls = {"n": 0}

    def junk(payload):
        calls["n"] += 1
        return "not json at all"

    client = _client(monkeypatch_chat=junk, max_retries=1)

    desc = client.describe(["uri"])

    assert calls["n"] == 2  # initial + 1 retry
    assert desc.parse_failed is True
    assert desc.interest is None


def test_describe_transport_error_raises_unavailable():
    from dcwb.vlm import VlmUnavailableError

    def boom(payload):
        raise OSError("connection refused")

    client = _client(monkeypatch_chat=boom)

    with pytest.raises(VlmUnavailableError):
        client.describe(["uri"])


def test_health_check_raises_unavailable_on_ping_error():
    from dcwb.vlm import VlmUnavailableError

    def boom():
        raise OSError("no route to host")

    client = _client(monkeypatch_ping=boom)

    with pytest.raises(VlmUnavailableError):
        client.health_check()


def test_build_payload_includes_schema_and_images_when_enabled():
    client = _client(monkeypatch_chat=lambda p: "{}", use_json_schema=True)

    payload = client.build_payload(["uri1", "uri2"])

    assert payload["model"] == "qwen2.5-vl-7b-instruct"
    assert payload["response_format"]["type"] == "json_schema"
    user = payload["messages"][1]["content"]
    image_parts = [p for p in user if p["type"] == "image_url"]
    assert len(image_parts) == 2


def test_build_payload_omits_schema_when_disabled():
    client = _client(monkeypatch_chat=lambda p: "{}", use_json_schema=False)

    payload = client.build_payload(["uri1"])

    assert "response_format" not in payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_vlm.py -v`
Expected: FAIL with `ImportError: cannot import name 'VlmClient'` (or `TypeError` on `VlmClient(...)`).

- [ ] **Step 3: Write minimal implementation**

Append to `src/dcwb/vlm.py`:

```python
import json
import urllib.request

_USER_INSTRUCTION = (
    "この前方カメラのフレーム列を見て、ドライブ記録のハイライトとしての魅力を評価してください。"
)
_JSON_HINT = (
    ' 次のJSONだけを返してください: '
    '{"interest": 0-10の整数, "scene_tags": [最大5個の短い語], '
    '"caption": "日本語1文", "drive_quality": "flowing|stop_and_go|stopped"}'
)


def _parse_description(content: str) -> ClipDescription | None:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    try:
        interest = max(0, min(10, int(data["interest"])))
    except Exception:
        return None
    tags = data.get("scene_tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t) for t in tags][:5]
    caption = str(data.get("caption") or "")
    dq = data.get("drive_quality")
    dq = dq if dq in _ALLOWED_DRIVE_QUALITY else ""
    return ClipDescription(
        interest=interest, scene_tags=tags, caption=caption, drive_quality=dq
    )


class VlmClient:
    """OpenAI-compatible VLM caller. The only network boundary.

    `chat` and `ping` are injectable for tests; defaults use urllib against the
    configured endpoint.
    """

    def __init__(self, config: VlmConfig, chat=None, ping=None):
        self.config = config
        self._chat = chat or self._http_chat
        self._ping = ping or self._http_ping

    def build_payload(self, frames_b64: list[str]) -> dict:
        cfg = self.config
        text = _USER_INSTRUCTION if cfg.use_json_schema else _USER_INSTRUCTION + _JSON_HINT
        user_content = [{"type": "text", "text": text}]
        for uri in frames_b64:
            user_content.append({"type": "image_url", "image_url": {"url": uri}})
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": cfg.effective_system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }
        if cfg.use_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "clip_description",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            }
        return payload

    def health_check(self) -> None:
        try:
            self._ping()
        except Exception as e:  # noqa: BLE001 - any failure means unavailable
            raise VlmUnavailableError(f"endpoint {self.config.endpoint} unreachable: {e}") from e

    def describe(self, frames_b64: list[str]) -> ClipDescription:
        payload = self.build_payload(frames_b64)
        for _ in range(self.config.max_retries + 1):
            try:
                content = self._chat(payload)
            except Exception as e:  # noqa: BLE001 - transport failure
                raise VlmUnavailableError(f"chat request failed: {e}") from e
            desc = _parse_description(content)
            if desc is not None:
                return desc
        return ClipDescription(
            interest=None, scene_tags=[], caption="", drive_quality="", parse_failed=True
        )

    def _http_chat(self, payload: dict) -> str:
        url = self.config.endpoint.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    def _http_ping(self) -> None:
        url = self.config.endpoint.rstrip("/") + "/models"
        req = urllib.request.Request(
            url, method="GET",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=min(10.0, self.config.timeout_sec)) as resp:
            resp.read()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_vlm.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/vlm.py tests/test_vlm.py
git commit -m "feat(vlm): add VlmClient with structured-output parse, retry, health check"
```

---

## Task 3: `highlight.py` AI scoring with per-clip cache

**Files:**
- Modify: `src/dcwb/highlight.py` (add imports, `AiScore`, `AiScoring`, frame sampling, cache helpers, `describe_candidates`, `_interest_to_total`)
- Test: `tests/test_highlight.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_highlight.py
class _FakeVlmClient:
    """Returns canned ClipDescriptions keyed by clip filename; counts calls."""

    def __init__(self, by_name, config):
        self.by_name = by_name
        self.config = config
        self.calls = 0

    def health_check(self):
        return None

    def describe(self, frames_b64):
        self.calls += 1
        # the most-recently sampled clip is tracked by the caller via _next_name
        return self.by_name[self._next_name]


def _ai_config(**kwargs):
    from dcwb.vlm import VlmConfig
    return VlmConfig.from_dict({"frames_per_clip": 2, "frame_max_edge": 64, **kwargs})


def test_describe_candidates_scores_by_interest_and_caches(tmp_path, monkeypatch):
    from dcwb import highlight
    from dcwb.highlight import HighlightCandidate, describe_candidates
    from dcwb.vlm import ClipDescription

    # two driving candidates over real synthetic clips (so frame sampling works)
    day = tmp_path / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    from tests.fixtures.make_synthetic import make_motion_clip
    clip_hi = day / "2026-05-08_00-00-00-front.mp4"
    clip_lo = day / "2026-05-08_00-01-00-front.mp4"
    make_motion_clip(clip_hi, duration_sec=1.0)
    make_motion_clip(clip_lo, duration_sec=1.0)
    cands = [
        HighlightCandidate(clip_hi, "2026-05-08_00-00-00", 1.0, SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 12.0, 8.0, 3.0, 10)),
        HighlightCandidate(clip_lo, "2026-05-08_00-01-00", 1.0, SegmentTelemetry(True, 10, {"DRIVE": 10}, True, 12.0, 8.0, 3.0, 10)),
    ]
    descs = {
        clip_hi.name: ClipDescription(9, ["海沿い"], "海沿いを流す", "flowing"),
        clip_lo.name: ClipDescription(2, ["渋滞"], "渋滞", "stop_and_go"),
    }

    client = _FakeVlmClient({}, _ai_config())
    # route each describe() to the clip currently being sampled
    real_sample = highlight._sample_frames_b64
    def tracking_sample(clip, duration, cfg):
        client._next_name = clip.name
        client.by_name = descs
        return real_sample(clip, duration, cfg)
    monkeypatch.setattr(highlight, "_sample_frames_b64", tracking_sample)

    cache_path = tmp_path / "vlm-cache.json"
    skips = []
    result = describe_candidates(cands, client, source_root=tmp_path, cache_path=cache_path, use_cache=True, skips=skips)

    assert result.calls == 2
    assert [s.candidate.clip.name for s in sorted(result.scores, key=lambda s: s.total, reverse=True)][0] == clip_hi.name
    assert cache_path.exists()

    # second run: cache hit, no new calls
    client2 = _FakeVlmClient(descs, _ai_config())
    monkeypatch.setattr(highlight, "_sample_frames_b64", tracking_sample)
    result2 = describe_candidates(cands, client2, source_root=tmp_path, cache_path=cache_path, use_cache=True, skips=[])
    assert client2.calls == 0
    assert result2.cache_hits == 2


def test_describe_candidates_records_skips(tmp_path, monkeypatch):
    from dcwb import highlight
    from dcwb.highlight import HighlightCandidate, describe_candidates
    from dcwb.vlm import ClipDescription

    day = tmp_path / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    from tests.fixtures.make_synthetic import make_motion_clip
    bad = day / "2026-05-08_00-00-00-front.mp4"
    low = day / "2026-05-08_00-01-00-front.mp4"
    make_motion_clip(bad, duration_sec=1.0)
    make_motion_clip(low, duration_sec=1.0)
    cands = [
        HighlightCandidate(bad, "2026-05-08_00-00-00", 1.0, SegmentTelemetry(True, 1, {"DRIVE": 1}, True, 1.0, 1.0, 0.0, 1)),
        HighlightCandidate(low, "2026-05-08_00-01-00", 1.0, SegmentTelemetry(True, 1, {"DRIVE": 1}, True, 1.0, 1.0, 0.0, 1)),
    ]
    descs = {
        bad.name: ClipDescription(None, [], "", "", parse_failed=True),
        low.name: ClipDescription(0, [], "暗い", "stopped"),  # below interest_min=1
    }
    client = _FakeVlmClient({}, _ai_config(interest_min=1))
    # stub sampling to return a dummy frame list and route describe() to this clip
    def sample_stub(clip, duration, cfg):
        client._next_name = clip.name
        client.by_name = descs
        return ["uri"]
    monkeypatch.setattr(highlight, "_sample_frames_b64", sample_stub)

    skips = []
    result = describe_candidates(cands, client, source_root=tmp_path, cache_path=tmp_path / "c.json", use_cache=False, skips=skips)

    assert result.scores == []
    reasons = {s["reason"] for s in skips}
    assert "vlm-parse-failed" in reasons
    assert "interest-below-min" in reasons
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_highlight.py -k describe_candidates -v`
Expected: FAIL with `ImportError: cannot import name 'describe_candidates'`.

- [ ] **Step 3: Write minimal implementation**

In `src/dcwb/highlight.py`, extend the imports at the top:

```python
from dcwb.ffmpeg_wrap import concat_clips, cut_clip, extract_frames, probe_duration
from dcwb.telemetry import SegmentTelemetry, read_segment_telemetry
from dcwb.vlm import PROMPT_VERSION, ClipDescription, VlmConfig, encode_frame
```

Then add (anywhere after the existing dataclasses, e.g. below `Excerpt`):

```python
@dataclass(frozen=True)
class AiScore:
    candidate: HighlightCandidate
    total: float
    description: ClipDescription
    cached: bool


@dataclass(frozen=True)
class AiScoring:
    scores: list[AiScore]
    calls: int
    cache_hits: int


def _frame_fractions(n: int) -> list[float]:
    if n <= 1:
        return [0.5]
    return [0.1 + 0.8 * i / (n - 1) for i in range(n)]


def _sample_frames_b64(clip: Path, duration_sec: float, cfg: VlmConfig) -> list[str]:
    if duration_sec <= 0:
        return []
    n = max(1, cfg.frames_per_clip)
    times = [duration_sec * f for f in _frame_fractions(n)]
    frames = extract_frames(clip, times)
    return [encode_frame(f, cfg.frame_max_edge) for f in frames]


def _cache_key(clip: Path) -> str:
    return f"{clip.name}:{clip.stat().st_mtime_ns}"


def load_vlm_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_vlm_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _interest_to_total(desc: ClipDescription) -> float:
    base = _clamp01((desc.interest or 0) / 10.0)
    if desc.drive_quality == "stopped":
        base = _clamp01(base - 0.3)
    return base


def describe_candidates(
    candidates: list[HighlightCandidate],
    vlm_client,
    source_root: Path,
    cache_path: Path,
    use_cache: bool = True,
    skips: list[dict] | None = None,
) -> AiScoring:
    cfg = vlm_client.config
    cache = load_vlm_cache(cache_path) if use_cache else {}
    calls = 0
    cache_hits = 0
    scores: list[AiScore] = []

    def record_skip(clip: Path, reason: str) -> None:
        if skips is not None:
            skips.append({"source_clip": clip.relative_to(source_root).as_posix(), "reason": reason})

    for cand in candidates:
        key = _cache_key(cand.clip)
        entry = cache.get(key) if use_cache else None
        if entry and entry.get("model") == cfg.model and entry.get("prompt_version") == PROMPT_VERSION:
            desc = ClipDescription(
                interest=entry.get("interest"),
                scene_tags=list(entry.get("scene_tags") or []),
                caption=entry.get("caption", ""),
                drive_quality=entry.get("drive_quality", ""),
            )
            cache_hits += 1
            cached = True
        else:
            frames = _sample_frames_b64(cand.clip, cand.duration_sec, cfg)
            if not frames:
                record_skip(cand.clip, "no-frames")
                continue
            desc = vlm_client.describe(frames)
            calls += 1
            cached = False
            if not desc.parse_failed and desc.interest is not None:
                cache[key] = {
                    "interest": desc.interest,
                    "scene_tags": desc.scene_tags,
                    "caption": desc.caption,
                    "drive_quality": desc.drive_quality,
                    "model": cfg.model,
                    "prompt_version": PROMPT_VERSION,
                }
        if desc.parse_failed or desc.interest is None:
            record_skip(cand.clip, "vlm-parse-failed")
            continue
        if desc.interest < cfg.interest_min:
            record_skip(cand.clip, "interest-below-min")
            continue
        scores.append(AiScore(candidate=cand, total=_interest_to_total(desc), description=desc, cached=cached))

    if use_cache:
        save_vlm_cache(cache_path, cache)
    return AiScoring(scores=scores, calls=calls, cache_hits=cache_hits)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_highlight.py -k describe_candidates -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/highlight.py tests/test_highlight.py
git commit -m "feat(highlight): add VLM interest scoring with per-clip cache"
```

---

## Task 4: `highlight.py` — AI branch and manifest in `highlight_day`

**Files:**
- Modify: `src/dcwb/highlight.py` (`_manifest_clip` gains `selection`; add `_manifest_clip_ai`; `highlight_day` gains `vlm_client`, `use_cache`, `selection`, plus the AI branch)
- Test: `tests/test_highlight.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_highlight.py
def test_highlight_day_ai_path_writes_manifest_with_ai_block(tmp_path, monkeypatch):
    import json
    from dcwb import highlight
    from dcwb.highlight import highlight_day
    from dcwb.vlm import ClipDescription
    from tests.fixtures.make_synthetic import make_motion_clip

    source = tmp_path / "usb"
    day = source / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    names = []
    for idx in range(2):
        clip = day / f"2026-05-08_00-0{idx}-00-front.mp4"
        make_motion_clip(clip, duration_sec=2.0)
        names.append(clip.name)
    monkeypatch.setattr(
        highlight, "read_segment_telemetry",
        lambda p: SegmentTelemetry(True, 60, {"DRIVE": 60}, True, 12.0, 8.0, 3.0, 60),
    )

    descs = {
        names[0]: ClipDescription(9, ["海沿い", "夕焼け"], "夕暮れの海岸を流す", "flowing"),
        names[1]: ClipDescription(7, ["市街地"], "街中を抜ける", "flowing"),
    }

    class FakeClient:
        config = _ai_config()
        calls = 0
        def health_check(self): return None
        def describe(self, frames_b64):
            type(self).calls += 1
            return descs[self._name]

    client = FakeClient()
    real_sample = highlight._sample_frames_b64
    def tracking_sample(clip, duration, cfg):
        client._name = clip.name
        return real_sample(clip, duration, cfg)
    monkeypatch.setattr(highlight, "_sample_frames_b64", tracking_sample)

    result = highlight_day(
        source_root=source, date="2026-05-08", out_root=tmp_path / "highlights",
        style="fast", allow_no_sei=False, encoder="libx264", bitrate_kbps=1000,
        target_duration_sec=1.0, vlm_client=client, use_cache=True,
    )

    assert result.output_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["ai_model"] == "qwen2.5-vl-7b-instruct"
    assert manifest["prompt_version"] == "1"
    assert manifest["vlm_calls"] == 2
    clip0 = manifest["clips"][0]
    assert clip0["selection"] == "ai"
    assert clip0["ai"]["interest"] in (7, 9)
    assert clip0["ai"]["scene_tags"]
    assert (tmp_path / "highlights" / "2026-05-08" / "vlm-cache.json").exists()


def test_highlight_day_mvp_path_marks_selection(tmp_path, monkeypatch):
    import json
    from dcwb import highlight
    from dcwb.highlight import highlight_day
    from tests.fixtures.make_synthetic import make_motion_clip

    source = tmp_path / "usb"
    day = source / "RecentClips" / "2026-05-08"
    day.mkdir(parents=True)
    make_motion_clip(day / "2026-05-08_00-00-00-front.mp4", duration_sec=2.0)
    monkeypatch.setattr(
        highlight, "read_segment_telemetry",
        lambda p: SegmentTelemetry(True, 60, {"DRIVE": 60}, True, 12.0, 8.0, 3.0, 60),
    )

    result = highlight_day(
        source_root=source, date="2026-05-08", out_root=tmp_path / "highlights",
        style="fast", allow_no_sei=False, encoder="libx264", bitrate_kbps=1000,
        target_duration_sec=1.0, vlm_client=None, selection="mvp-fallback",
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["clips"][0]["selection"] == "mvp-fallback"
    assert manifest["clips"][0]["scores"]  # MVP component scores still present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_highlight.py -k "ai_path or mvp_path" -v`
Expected: FAIL (`highlight_day() got an unexpected keyword argument 'vlm_client'`).

- [ ] **Step 3: Write minimal implementation**

In `src/dcwb/highlight.py`, change `_manifest_clip` to accept and record a selection label:

```python
def _manifest_clip(excerpt: Excerpt, source_root: Path, rendered_path: Path, selection: str) -> dict:
    scored = excerpt.source
    candidate = scored.candidate
    tel = candidate.telemetry
    return {
        "source_clip": candidate.clip.relative_to(source_root).as_posix(),
        "rendered_clip": rendered_path.name,
        "start_sec": round(excerpt.start_sec, 3),
        "duration_sec": round(excerpt.duration_sec, 3),
        "selection": selection,
        "score": round(scored.total, 4),
        "scores": {key: round(value, 4) for key, value in scored.components.items()},
        "visual": {
            "mean_luma": round(scored.visual.mean_luma, 4),
            "visual_change": round(scored.visual.visual_change, 4),
        },
        "telemetry": {
            "has_sei": tel.has_sei,
            "gear_counts": tel.gear_counts,
            "max_speed_mps": round(tel.max_speed_mps, 4),
            "avg_speed_mps": round(tel.avg_speed_mps, 4),
            "speed_delta_mps": round(tel.speed_delta_mps, 4),
        },
        "low_confidence": candidate.low_confidence,
    }
```

Add the AI manifest writer next to it:

```python
def _manifest_clip_ai(excerpt: Excerpt, source_root: Path, rendered_path: Path, model: str) -> dict:
    scored = excerpt.source  # AiScore
    candidate = scored.candidate
    tel = candidate.telemetry
    desc = scored.description
    return {
        "source_clip": candidate.clip.relative_to(source_root).as_posix(),
        "rendered_clip": rendered_path.name,
        "start_sec": round(excerpt.start_sec, 3),
        "duration_sec": round(excerpt.duration_sec, 3),
        "selection": "ai",
        "score": round(scored.total, 4),
        "ai": {
            "interest": desc.interest,
            "scene_tags": desc.scene_tags,
            "caption": desc.caption,
            "drive_quality": desc.drive_quality,
            "model": model,
            "cached": scored.cached,
        },
        "telemetry": {
            "has_sei": tel.has_sei,
            "gear_counts": tel.gear_counts,
            "max_speed_mps": round(tel.max_speed_mps, 4),
            "avg_speed_mps": round(tel.avg_speed_mps, 4),
            "speed_delta_mps": round(tel.speed_delta_mps, 4),
        },
        "low_confidence": candidate.low_confidence,
    }
```

Replace the `highlight_day` function with this version (adds `vlm_client`, `use_cache`, `selection`, the AI branch, and threads `selection` into the MVP manifest writer):

```python
def highlight_day(
    source_root: Path,
    date: str,
    out_root: Path,
    style: str = "fast",
    allow_no_sei: bool = False,
    encoder: str = "h264_videotoolbox",
    bitrate_kbps: int = 12000,
    target_duration_sec: float | None = None,
    vlm_client=None,
    use_cache: bool = True,
    selection: str = "mvp",
) -> HighlightResult:
    clips = discover_day_front_clips(source_root, date)
    skips: list[dict] = []
    candidates = build_candidates(clips, allow_no_sei=allow_no_sei, skips=skips)
    day_out = out_root / date
    clip_out = day_out / "clips"
    day_out.mkdir(parents=True, exist_ok=True)
    manifest_path = day_out / f"highlight-{style}.json"
    output_path = day_out / f"highlight-{style}.mp4"

    use_ai = vlm_client is not None
    ai_meta = {}
    if use_ai:
        scoring = describe_candidates(
            candidates, vlm_client, source_root,
            cache_path=day_out / "vlm-cache.json", use_cache=use_cache, skips=skips,
        )
        scores = scoring.scores
        cfg = vlm_client.config
        ai_meta = {
            "ai_endpoint": cfg.endpoint,
            "ai_model": cfg.model,
            "prompt_version": PROMPT_VERSION,
            "vlm_calls": scoring.calls,
            "vlm_cache_hits": scoring.cache_hits,
        }
    else:
        scores = score_candidates(candidates)

    if not scores:
        manifest = {
            "date": date,
            "style": style,
            "source": str(source_root),
            "created_at": datetime.now(JST).isoformat(),
            "target_duration_sec": target_duration_sec or STYLE_CONFIGS[style].target_sec,
            "output": output_path.name,
            "clips": [],
            "skips": skips or [{"reason": "no eligible driving front clips"}],
            **ai_meta,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        return HighlightResult(output_path, manifest_path, [], 0)

    excerpts = plan_excerpts(scores, style, target_duration_sec=target_duration_sec)
    rendered: list[Path] = []
    manifest_clips: list[dict] = []
    for idx, excerpt in enumerate(excerpts, start=1):
        rendered_path = clip_out / f"{idx:03d}-{excerpt.ts_str}.mp4"
        cut_clip(
            excerpt.source.candidate.clip, rendered_path,
            excerpt.start_sec, excerpt.duration_sec,
            encoder=encoder, bitrate_kbps=bitrate_kbps,
        )
        rendered.append(rendered_path)
        if use_ai:
            manifest_clips.append(_manifest_clip_ai(excerpt, source_root, rendered_path, vlm_client.config.model))
        else:
            manifest_clips.append(_manifest_clip(excerpt, source_root, rendered_path, selection))
    if rendered:
        concat_clips(rendered, output_path, encoder=encoder, bitrate_kbps=bitrate_kbps)
    manifest = {
        "date": date,
        "style": style,
        "source": str(source_root),
        "created_at": datetime.now(JST).isoformat(),
        "target_duration_sec": target_duration_sec or STYLE_CONFIGS[style].target_sec,
        "output": output_path.name,
        "clips": manifest_clips,
        "skips": skips,
        **ai_meta,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return HighlightResult(output_path, manifest_path, rendered, len(rendered))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_highlight.py -v`
Expected: PASS — new AI/MVP tests plus all pre-existing highlight tests (the existing `test_highlight_day_writes_video_and_manifest` still passes; `_manifest_clip` now also emits `selection` but keeps `scores`).

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/highlight.py tests/test_highlight.py
git commit -m "feat(highlight): AI selection branch and manifest in highlight_day"
```

---

## Task 5: `cli.py` — flags, config load, health-check/fallback wiring

**Files:**
- Modify: `src/dcwb/cli.py` (imports, `highlight-day` parser args, `_cmd_highlight_day`)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_cli.py
def test_highlight_day_hard_errors_when_vlm_unavailable(tmp_path, monkeypatch, capsys):
    from dcwb import cli
    from dcwb.vlm import VlmUnavailableError

    class DeadClient:
        def __init__(self, config): self.config = config
        def health_check(self): raise VlmUnavailableError("refused")

    called = {"n": 0}
    def fake_highlight_day(**kwargs):
        called["n"] += 1
    monkeypatch.setattr(cli, "VlmClient", DeadClient)
    monkeypatch.setattr(cli, "highlight_day", fake_highlight_day)

    rc = cli.main([
        "highlight-day", "--source", str(tmp_path), "--date", "2026-05-08",
        "--out-root", str(tmp_path / "out"), "--pipeline-config", str(tmp_path / "missing.json"),
    ])

    assert rc == 1
    assert called["n"] == 0  # never started work
    assert "VLM unavailable" in capsys.readouterr().err


def test_highlight_day_falls_back_with_allow_no_ai(tmp_path, monkeypatch):
    from dcwb import cli
    from dcwb.vlm import VlmUnavailableError
    from dcwb.highlight import HighlightResult

    class DeadClient:
        def __init__(self, config): self.config = config
        def health_check(self): raise VlmUnavailableError("refused")

    captured = {}
    def fake_highlight_day(**kwargs):
        captured.update(kwargs)
        return HighlightResult(tmp_path / "h.mp4", tmp_path / "h.json", [tmp_path / "h.mp4"], 1)
    monkeypatch.setattr(cli, "VlmClient", DeadClient)
    monkeypatch.setattr(cli, "highlight_day", fake_highlight_day)

    rc = cli.main([
        "highlight-day", "--source", str(tmp_path), "--date", "2026-05-08",
        "--out-root", str(tmp_path / "out"), "--allow-no-ai",
        "--pipeline-config", str(tmp_path / "missing.json"),
    ])

    assert rc == 0
    assert captured["vlm_client"] is None
    assert captured["selection"] == "mvp-fallback"


def test_highlight_day_passes_client_when_healthy(tmp_path, monkeypatch):
    from dcwb import cli
    from dcwb.highlight import HighlightResult

    class LiveClient:
        def __init__(self, config): self.config = config
        def health_check(self): return None

    captured = {}
    def fake_highlight_day(**kwargs):
        captured.update(kwargs)
        return HighlightResult(tmp_path / "h.mp4", tmp_path / "h.json", [tmp_path / "h.mp4"], 1)
    monkeypatch.setattr(cli, "VlmClient", LiveClient)
    monkeypatch.setattr(cli, "highlight_day", fake_highlight_day)

    rc = cli.main([
        "highlight-day", "--source", str(tmp_path), "--date", "2026-05-08",
        "--out-root", str(tmp_path / "out"), "--vlm-model", "custom-vlm",
        "--pipeline-config", str(tmp_path / "missing.json"),
    ])

    assert rc == 0
    assert isinstance(captured["vlm_client"], LiveClient)
    assert captured["vlm_client"].config.model == "custom-vlm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_cli.py -k highlight_day -v`
Expected: FAIL (`AttributeError: module 'dcwb.cli' has no attribute 'VlmClient'`).

- [ ] **Step 3: Write minimal implementation**

In `src/dcwb/cli.py`, extend imports near the existing `from dcwb.highlight import highlight_day`:

```python
from dcwb.highlight import highlight_day
from dcwb.vlm import VlmClient, VlmConfig, VlmUnavailableError
```

Add `from dataclasses import replace` to the top-level imports (next to the other stdlib imports).

In the `highlight-day` parser block (after `ph.add_argument("--bitrate-kbps", ...)`), add:

```python
    ph.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CFG)
    ph.add_argument("--vlm-endpoint", default=None, help="Override highlight_ai.endpoint")
    ph.add_argument("--vlm-model", default=None, help="Override highlight_ai.model")
    ph.add_argument("--allow-no-ai", action="store_true", help="Fall back to the MVP scorer if the VLM is unavailable")
    ph.add_argument("--no-vlm-cache", action="store_true", help="Ignore and overwrite the per-day VLM cache")
```

Replace `_cmd_highlight_day` with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_cli.py -k highlight_day -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dcwb/cli.py tests/test_cli.py
git commit -m "feat(cli): wire highlight-day to VLM with health check and --allow-no-ai fallback"
```

---

## Task 6: `pipeline.json`, docs, and full verification

**Files:**
- Modify: `pipeline.json`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the `highlight_ai` config section**

Edit `pipeline.json` to add a `highlight_ai` block as a sibling of `awb` and `prune`:

```json
  "highlight_ai": {
    "endpoint": "http://localhost:1234/v1",
    "model": "qwen2.5-vl-7b-instruct",
    "api_key": "lm-studio",
    "frames_per_clip": 3,
    "frame_max_edge": 512,
    "timeout_sec": 120,
    "max_retries": 1,
    "interest_min": 1,
    "system_prompt": null,
    "use_json_schema": true
  }
```

- [ ] **Step 2: Verify the config parses and round-trips through VlmConfig**

Run:
```bash
uv run python -c "import json; from dcwb.vlm import VlmConfig; c=VlmConfig.from_dict(json.load(open('pipeline.json'))['highlight_ai']); print(c.model, c.frames_per_clip, c.use_json_schema)"
```
Expected: `qwen2.5-vl-7b-instruct 3 True`

- [ ] **Step 3: Document the AI path in CLAUDE.md**

In `CLAUDE.md`, update the `highlight-day` paragraph (the one beginning "`highlight-day` は `RecentClips/<date>`...") to note the new default behavior. Append:

```markdown
`highlight-day` の選定は **VLM 主導**: テレメトリは走行判定（DRIVE/REVERSE を含むか）だけに使い、各 front クリップから代表数フレーム（既定 3 枚、長辺 512px）を抽出して LM Studio などの OpenAI 互換エンドポイント（`pipeline.json` の `highlight_ai`、既定 `http://localhost:1234/v1`）に送り、`{interest 0-10, scene_tags, caption, drive_quality}` を構造化出力で受け取る。`interest` 上位で `fast`/`cruise` のターゲット尺を充填し、caption/tags は manifest にのみ記録（焼き込みなし）。VLM の I/O は `vlm.py` に隔離。VLM 結果は `highlights/<date>/vlm-cache.json` にクリップ単位でキャッシュ（`model`/`prompt_version` 込みで失効、2スタイル共有）。**エンドポイント不通時は作業前に即停止**し、`--allow-no-ai` を付けたときだけ従来の MVP スコア（SEI＋OpenCV）にフォールバックして manifest に `selection: "mvp-fallback"` を残す。`--vlm-endpoint`/`--vlm-model` で config を上書き、`--no-vlm-cache` でキャッシュ無視。
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run --extra dev pytest`
Expected: PASS — all suites green, including the untouched MVP highlight tests and the new `tests/test_vlm.py`.

- [ ] **Step 5: Commit**

```bash
git add pipeline.json CLAUDE.md
git commit -m "feat(highlight): add highlight_ai config and document the VLM path"
```

---

## Self-Review

**Spec coverage:**
- AI-driven selection (telemetry only gates drove-vs-parked) → Task 3 `describe_candidates` over `build_candidates` output; Task 4 AI branch. ✓
- Structured output `{interest, scene_tags, caption, drive_quality}` → Task 1 `RESPONSE_SCHEMA`, Task 2 `_parse_description`. ✓
- Two styles, fill by interest → Task 3 `_interest_to_total`, reuse of `plan_excerpts` in Task 4. ✓
- Captions in manifest only → Task 4 `_manifest_clip_ai` (no ffmpeg overlay anywhere). ✓
- Per-clip cache shared by fast/cruise, invalidated by model/prompt_version → Task 3 cache helpers + validity check; Task 4 writes `vlm-cache.json` in `day_out` (shared by both styles). ✓
- Fail loud unless `--allow-no-ai` → Task 5 health check before work; hard error vs fallback. ✓
- Frame sampling: 3 frames at 10/50/90%, 512px, one call → Task 1 `encode_frame`, Task 3 `_frame_fractions`/`_sample_frames_b64`, Task 2 multi-image payload. ✓
- `use_json_schema` fallback to prompt-guided parse → Task 2 `build_payload` branch + lenient `_parse_description`. ✓
- Manifest top-level `ai_endpoint`/`ai_model`/`prompt_version`/`vlm_calls`/`vlm_cache_hits`; skips `vlm-parse-failed`/`interest-below-min` → Task 4 `ai_meta`; Task 3 `record_skip`. ✓
- Per-clip parse failure drops only that clip → Task 3 `vlm-parse-failed` skip + continue. ✓
- MVP scorer untouched, existing tests unchanged → MVP functions not modified; `_manifest_clip` only gains an additive `selection` field (existing assertions check `scores`, still present). ✓
- Config in `pipeline.json` `highlight_ai`, `--vlm-endpoint`/`--vlm-model` overrides → Task 5, Task 6. ✓

**Placeholder scan:** None. All steps include full code and exact commands. (An earlier dead stub line in Task 3's second test was removed inline.)

**Type consistency:** `VlmConfig`, `ClipDescription`, `VlmUnavailableError`, `VlmClient(config, chat=, ping=)`, `describe`, `health_check`, `build_payload`, `encode_frame`, `PROMPT_VERSION` are used identically across `vlm.py`, `highlight.py`, `cli.py`, and tests. `AiScore`/`AiScoring` field names (`scores`, `calls`, `cache_hits`, `candidate`, `total`, `description`, `cached`) match between definition (Task 3) and use (Task 4). `highlight_day` keyword args (`vlm_client`, `use_cache`, `selection`) match between definition (Task 4) and callers (Task 5, tests). `plan_excerpts` is duck-typed over `.total`/`.candidate`, which `AiScore` provides.
