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
    scene_tags: list[str]
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
    def from_dict(cls, data: dict | None) -> VlmConfig:
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
