from __future__ import annotations

import base64
import dataclasses
import json
import urllib.request
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
    # Sampling params sent in every request. LM Studio's UI sampling settings do
    # NOT reach the OpenAI-compatible API, so these must be sent explicitly.
    # max_tokens=300 truncated generation mid-stream and triggered repetition
    # loops; a higher cap plus repeat_penalty lets the model stop naturally.
    max_tokens: int = 768
    temperature: float = 0.2
    repeat_penalty: float = 1.4
    frequency_penalty: float = 0.5

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
        lines = text.splitlines()
        inner = [ln for ln in lines[1:] if ln.strip() != "```"]
        text = "\n".join(inner)
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
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "repeat_penalty": cfg.repeat_penalty,
            "frequency_penalty": cfg.frequency_penalty,
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
