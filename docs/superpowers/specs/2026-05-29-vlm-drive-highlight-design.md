# VLM-driven drive highlight selection -- design

Date: 2026-05-29
Status: approved (brainstorming)

## Background / motivation

`dcwb highlight-day` already builds daily front-camera drive highlights from
`RecentClips/<YYYY-MM-DD>`, but its MVP scores segments only from cheap local
signals: Tesla SEI telemetry (gear/speed) plus OpenCV brightness and frame
change. It has no idea *what* is on screen. The MVP design explicitly listed
object detection / scene classification / CLIP scoring as non-goals "to be added
later once the daily front-camera pipeline is stable."

This is that later step: use an open-weight Vision-Language Model (VLM) to both
**understand scene content for selection** and **generate a short description**
per excerpt. The VLM runs on a separate LAN GPU machine, reached over an
OpenAI-compatible HTTP API; `dcwb` itself stays where it already runs.

## Goals

- AI-driven selection: telemetry decides only *drove vs parked*; a VLM judges
  *how good a highlight* each driving clip is.
- Per-clip structured VLM output: `{interest, scene_tags, caption, drive_quality}`.
- Keep both existing styles (`fast`, `cruise`); selection fills the style's
  target duration by `interest` rank.
- Record VLM output (caption, tags, interest) in the manifest only -- no
  burned-in subtitles.
- Cache VLM results per clip so re-runs and the two styles share one round of
  calls (cost control).
- Fail loud: if the VLM endpoint is unreachable, stop before doing work, unless
  `--allow-no-ai` explicitly permits falling back to the MVP scorer.

## Non-goals

- Burning captions / subtitles into the video (manifest only).
- Running the model on the Mac or this Linux box (it lives on a LAN GPU host).
- 6-camera grid, music/beat matching, serve UI integration.
- Replacing or removing the existing MVP scorer (it survives as the fallback).
- Multi-image temporal reasoning beyond a few sampled frames per clip.

## Environment / deployment

- Inference host: Windows PC, GeForce RTX 5070Ti 16GB, running **LM Studio** with
  its OpenAI-compatible server (default `http://<host>:1234/v1`).
- Default model: a 7-8B GGUF VLM, e.g. `qwen2.5-vl-7b-instruct` (fits 16GB
  comfortably; configurable).
- `dcwb` runs as today (Mac for real use, Linux for tests); recordings live on
  QNAP. The VLM is just an HTTP dependency.

## Architecture

New module **`src/dcwb/vlm.py`** isolates the VLM I/O boundary, in the same
spirit as `ffmpeg_wrap` and `telemetry`.

```
build_candidates                (existing: telemetry passes only driving clips)
        |  driving clips
        v
score_candidates_ai             (new path, driven by vlm.py)
   |- per clip: extract_frames -> N representative frames (existing ffmpeg_wrap)
   |- look up vlm-cache.json (key = clip name + mtime) -> hit skips the call
   |- miss: VlmClient.describe_clip(frames) -> {interest, scene_tags, caption,
   |                                            drive_quality}
   |- update cache
        |  candidates carrying interest scores
        v
plan_excerpts                   (existing: top interest fills target, fast/cruise)
        v
cut_clip -> concat_clips        (existing) + manifest records VLM output
```

### Responsibilities

- **`vlm.py`** — OpenAI-compatible client, frame base64 encoding, JSON-schema
  enforced output, retry, and raising `VlmUnavailableError` on failure. Holds no
  selection logic.
- **`highlight.py`** — attaches VLM results to candidates, scores by `interest`,
  manages the cache, plans excerpts. Falls back to the existing MVP scorer only
  when `--allow-no-ai` is set.
- **`cli.py`** — thin wiring; adds `--vlm-endpoint`, `--vlm-model`,
  `--allow-no-ai`, `--no-vlm-cache`.

The MVP scorer (`score_candidate` / `score_candidates`) is kept unchanged as the
fallback path, so existing tests stay valid.

## Frame sampling

- Reuse `extract_frames(clip, times)`.
- Per clip (~60s): sample **3 frames** at 10% / 50% / 90% of duration
  (`frames_per_clip`).
- Resize each to a long edge of **512px** (`frame_max_edge`), JPEG quality 85,
  base64. Send all frames in one VLM call (Qwen2.5-VL supports multi-image),
  balancing VRAM/token cost against capturing the clip's flow.

## VLM output schema

Enforced via OpenAI-compatible `response_format: json_schema`:

```json
{
  "interest": 0,
  "scene_tags": ["..."],
  "caption": "...",
  "drive_quality": "flowing|stop_and_go|stopped"
}
```

- `interest`: integer 0-10, the primary selection score ("how pleasant to
  rewatch as a drive memory").
- `scene_tags`: up to 5 short tags (e.g. "海沿い", "夕焼け", "トンネル", "市街地",
  "渋滞").
- `caption`: one short Japanese sentence.
- `drive_quality`: auxiliary signal; `"stopped"` is penalized so even AI-driven
  selection reliably drops stationary clips.

If LM Studio does not honor `json_schema`, fall back to a prompt instructing JSON
output plus a lenient parse and one retry. A clip whose output still fails to
parse gets `interest = null` (excluded from selection; recorded in `skips`).

### Prompt

System prompt fixes the taste: highly rate footage pleasant to rewatch later as
a drive record -- flowing scenery, distinctive landscapes, changing townscapes;
low-rate monotonous / stopped / very dark footage. Overridable via config
(`system_prompt`). A `prompt_version` string is bumped when the built-in prompt
changes, to invalidate the cache.

## Configuration

New `highlight_ai` section in `pipeline.json`:

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

- `--vlm-endpoint` / `--vlm-model` override the config, matching the existing
  `--pipeline-config` style.
- `interest_min`: clips scoring below this are excluded.
- `system_prompt: null` uses the built-in default.

## Cache

File `highlights/<date>/vlm-cache.json`:

- Key: `<clip_name>:<mtime_ns>`.
- Value: raw VLM output (`interest` / `scene_tags` / `caption` / `drive_quality`)
  plus `model` and `prompt_version`.
- An entry is valid only if key, `model`, and `prompt_version` all match;
  changing the model or built-in prompt auto-invalidates and re-fetches.
- `fast` and `cruise` share the same day's cache, so the VLM is called at most
  once per clip per day. `--no-vlm-cache` ignores the cache.

## Health check

Before any work, ping the endpoint. If unreachable and `--allow-no-ai` is not
set, raise `VlmUnavailableError` and stop **before** extracting frames or running
ffmpeg (no wasted work, non-zero exit, clear message).

## Manifest changes

Each clip entry in `highlight-<style>.json` gains:

```json
{
  "source_clip": "...",
  "rendered_clip": "...",
  "start_sec": 0.0,
  "duration_sec": 0.0,
  "selection": "ai",
  "ai": {
    "interest": 8,
    "scene_tags": ["海沿い", "夕焼け"],
    "caption": "夕暮れの海岸沿いを流す",
    "drive_quality": "flowing",
    "model": "qwen2.5-vl-7b-instruct",
    "cached": true
  },
  "telemetry": { "...": "existing" }
}
```

- `selection`: `"ai"` or `"mvp-fallback"`.
- Top-level: `ai_endpoint`, `ai_model`, `prompt_version`, `vlm_calls` (actual
  calls made), `vlm_cache_hits` -- for reproducibility and cost visibility.
- `skips` records AI-origin exclusions: `reason: "vlm-parse-failed"`,
  `"interest-below-min"`.

## Error handling

- Endpoint unreachable + no `--allow-no-ai` -> `VlmUnavailableError` before work
  starts (non-zero exit, clear message).
- `--allow-no-ai` set + unreachable -> fall back to the MVP scorer, manifest
  `selection: "mvp-fallback"`, warning to stderr.
- Per-clip parse failure -> drop only that clip, continue (one bad clip never
  fails the whole run).

## Testing

In `tests/`, following the existing synthetic-mp4 + libx264 convention.

- Make `VlmClient` an injectable protocol; tests inject a fake returning canned
  JSON (no real endpoint needed).
- Cases:
  1. AI path selects clips in `interest` order.
  2. Cache hit: second run makes no VLM call.
  3. Unreachable without `--allow-no-ai`: raises, no output file produced.
  4. `--allow-no-ai` falls back; manifest `selection: "mvp-fallback"`.
  5. Parse-failure clip is recorded in `skips`; others are still selected.
  6. `use_json_schema: false` exercises prompt-guided parsing.
- Existing MVP tests remain unchanged (fallback path stays alive).
