from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest


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


def test_parse_description_clamps_and_coerces_interest():
    from dcwb.vlm import _parse_description

    over = _parse_description('{"interest": 13, "scene_tags": [], "caption": "x", "drive_quality": "flowing"}')
    assert over.interest == 10

    as_str = _parse_description('{"interest": "8", "scene_tags": [], "caption": "x", "drive_quality": "flowing"}')
    assert as_str.interest == 8


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

    assert calls["n"] == 2
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
