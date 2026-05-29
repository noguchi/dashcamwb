from __future__ import annotations

import base64

import cv2
import numpy as np


def test_vlm_config_from_dict_merges_defaults_and_ignores_unknown():
    from dcwb.vlm import VlmConfig

    cfg = VlmConfig.from_dict({"model": "my-vlm", "frames_per_clip": 5, "bogus": 1})

    assert cfg.model == "my-vlm"
    assert cfg.frames_per_clip == 5
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
    frame[:, :, 0] = 200

    uri = encode_frame(frame, max_edge=512)

    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 512
