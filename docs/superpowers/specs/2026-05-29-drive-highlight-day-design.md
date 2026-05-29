# Drive highlight day videos -- design

Date: 2026-05-29
Status: approved (brainstorming)

## Background / motivation

`dcwb` already corrects Tesla DashCam color and can read Tesla SEI telemetry from
front-camera mp4 files. The next useful workflow is a daily "drive memory"
highlight: not accident detection, but a short video that is pleasant to rewatch.

The first version should prefer reliable local signals over heavyweight AI. Good
segments are moving, visually changing, and readable. Bad segments are stopped,
dark, monotonous, or dominated by long waits.

## Goals

- Build a daily highlight from `RecentClips/<YYYY-MM-DD>`.
- Use only the `front` camera for the first version.
- Support two styles:
  - `fast`: short 8-15 second excerpts, more cuts, higher tempo.
  - `cruise`: longer 30-60 second excerpts, fewer cuts, stronger drive feel.
- Use local telemetry and frame analysis only: no external service and no AI
  model dependency in the MVP.
- Produce a manifest explaining why each excerpt was selected.

## Non-goals

- 6-camera grid output.
- Automatic music selection or beat matching.
- Object detection, scene classification, or CLIP scoring.
- Serve UI integration.
- Highlighting dangerous events specifically.

These can be added later once the daily front-camera pipeline is stable.

## CLI

```
uv run dcwb highlight-day \
  --source /mnt/sentryusb \
  --date 2026-05-08 \
  --style fast

uv run dcwb highlight-day \
  --source /mnt/sentryusb \
  --date 2026-05-08 \
  --style cruise
```

Defaults:

- `--source`: `/Volumes/sentryusb`, matching other commands.
- `--out-root`: `highlights`.
- `--style`: `fast`.
- `--allow-no-sei`: false. Without SEI, a clip is skipped in the MVP because the
  tool cannot reliably prove the car was driving. This may be relaxed later.

Output:

```
highlights/YYYY-MM-DD/highlight-fast.mp4
highlights/YYYY-MM-DD/highlight-fast.json
highlights/YYYY-MM-DD/clips/*.mp4
```

The intermediate `clips/` files make debugging and manual review easier.

## Input Model

Input files are front-camera RecentClips:

```
<source>/RecentClips/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS-front.mp4
```

Each one-minute front clip becomes a candidate segment. A segment is eligible
only when:

- the file is readable,
- telemetry has SEI,
- `gear_state` contains `DRIVE` or `REVERSE`,
- the clip duration is positive.

No-SEI clips are skipped by default. With `--allow-no-sei`, they may be scored by
visual signals only, but this is explicitly lower confidence and should be
marked in the manifest.

## Scoring

Each eligible segment receives a transparent weighted score.

Positive signals:

- `speed_score`: higher average speed is more drive-like.
- `speed_delta_score`: acceleration and deceleration create tempo.
- `visual_change_score`: frame-to-frame visual change suggests movement through
  changing scenery.
- `brightness_score`: mid-to-bright clips are easier to watch.

Penalties:

- `still_penalty`: long stopped or nearly stopped periods.
- `dark_penalty`: very dark clips.
- `monotony_penalty`: low visual change across sampled frames.

The MVP should keep the formula simple and deterministic. The manifest stores
all component scores so thresholds and weights can be tuned from real output.

## Style Selection

`fast`:

- excerpt length: 8-15 seconds,
- target duration: 180 seconds,
- choose more segments,
- prefer high `speed_delta_score` and high `visual_change_score`,
- avoid selecting adjacent segments unless both score highly.

`cruise`:

- excerpt length: 30-60 seconds,
- target duration: 360 seconds,
- choose fewer segments,
- prefer sustained `speed_score`, readable brightness, and moderate visual
  change,
- allow adjacent segments to merge into longer excerpts.

Both styles should deduplicate overlapping excerpts and preserve chronological
order in the final video.

## Rendering

The MVP can render excerpts directly from source clips with ffmpeg:

1. score all eligible front clips for the day,
2. select excerpt windows according to style,
3. cut each excerpt to `highlights/YYYY-MM-DD/clips/`,
4. concat excerpts into `highlight-<style>.mp4`,
5. write `highlight-<style>.json`.

Color correction is not required in the first pass. A later version can reuse the
existing render pipeline to apply profiles before concat.

## Manifest

`highlight-<style>.json` contains:

```
{
  "date": "2026-05-08",
  "style": "fast",
  "source": "/mnt/sentryusb",
  "created_at": "2026-05-29T12:00:00+09:00",
  "target_duration_sec": 180,
  "output": "highlight-fast.mp4",
  "clips": [
    {
      "source_clip": "RecentClips/2026-05-08/2026-05-08_07-14-49-front.mp4",
      "start_sec": 12.0,
      "duration_sec": 12.0,
      "score": 0.82,
      "scores": {
        "speed": 0.31,
        "speed_delta": 0.18,
        "visual_change": 0.24,
        "brightness": 0.12,
        "penalty": -0.03
      },
      "telemetry": {
        "has_sei": true,
        "gear_counts": {"DRIVE": 1700, "PARK": 80},
        "max_speed_mps": 18.5
      }
    }
  ]
}
```

## Module Boundaries

New module: `src/dcwb/highlight.py`

- segment discovery for a day,
- telemetry eligibility checks,
- frame sampling features,
- score calculation,
- excerpt selection,
- manifest generation.

CLI changes stay in `src/dcwb/cli.py` and only parse arguments, call
`highlight_day`, and report output paths.

ffmpeg operations should reuse helpers in `ffmpeg_wrap.py` where practical, but
excerpt cutting and concat may need small new helpers because they are different
from matrix rendering.

## Error Handling

- Missing day directory: return a clear error and non-zero exit.
- No eligible clips: write no video, print a clear message, return zero.
- Individual corrupt clips: skip and record the skip reason in the manifest.
- ffmpeg failure during excerpt creation or concat: fail the command and keep
  intermediate files for inspection.

## Testing

Use TDD for implementation.

Unit tests:

- discover only `*-front.mp4` clips for the requested date,
- skip no-SEI clips by default,
- include no-SEI clips only with `allow_no_sei`,
- score brighter/changing/moving clips above dark/static/stopped clips,
- produce different excerpt plans for `fast` and `cruise`,
- manifest includes score components and telemetry summary.

CLI tests:

- `highlight-day --date YYYY-MM-DD --style fast` calls the highlight module with
  resolved paths,
- invalid style is rejected by argparse,
- missing day directory returns non-zero.

Integration-style test:

- create synthetic mp4 clips, render a short highlight with the test encoder,
  and assert the output mp4 and manifest exist.
