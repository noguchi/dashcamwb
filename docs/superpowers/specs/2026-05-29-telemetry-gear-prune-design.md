# Telemetry (gear) based RecentClips pruning — design

Date: 2026-05-29
Status: approved (brainstorming)

## Background / motivation

`prune-recent` currently classifies RecentClips segments as low-value using a
**pixel motion score** (front-camera frame-diff) — see
[2026-05-28-recentclips-low-motion-prune-design.md](2026-05-28-recentclips-low-motion-prune-design.md).
That proxy is noisy: a parked car watching busy traffic scores high (wrongly
kept), and slow/monotonous driving can score low (wrongly quarantined).

Tesla embeds true vehicle telemetry directly in each dashcam mp4 as **SEI
(Supplemental Enhancement Information) NAL units inside the H.264 stream, one per
frame, protobuf-encoded**. This includes `gear_state` (PARK/DRIVE/REVERSE/
NEUTRAL), speed, and GPS. Reading the gear lets us decide "was the car actually
driving" without any pixel heuristic, fully locally, and retroactively on
existing footage.

Documented officially at **github.com/teslamotors/dashcam** (`dashcam.proto`,
`sei_extractor.py`, `sei_explorer.html`). Requirements: firmware 2025.44.25+ and
HW3+. **If the car is parked, SEI may be absent** — so SEI-absence is ambiguous
(parked-on-supported-firmware OR unsupported-firmware).

### Verified facts (spike, 2026-05-29)

- SEI parsed from a real driving clip yields gear `PARK 147 / DRIVE 2014`,
  exactly matching the independently-found `drive-data.json` run-lengths.
- The protobuf payload sits after a 4-byte `42 42 42 69` ("BBBi") marker inside
  each SEI user_data_unregistered NAL; `08 01` (version=1) follows.
- `gear_state` enum: `PARK=0, DRIVE=1, REVERSE=2, NEUTRAL=3`.
- SEI coverage on this USB is mixed: present on driving clips from ~2026-05-06
  onward, absent on parked clips and older footage → motion fallback is
  mandatory, not optional.
- Official `sei_extractor.py` runs in the uv env with only `protobuf` (runtime)
  + `grpcio-tools` (to compile the proto), giving the same gear result.

## Goals

- Use real `gear_state` as the primary "driving vs parked" signal for prune.
- Keep the existing motion score as a fallback when SEI is absent.
- Preserve existing age / event-overlap protection guards unchanged.
- Backward compatible: a config flag restores pure-motion behaviour.

## Non-goals (YAGNI)

- Showing telemetry in the serve UI.
- Persisting telemetry into `_pipeline.json`.
- Speed-threshold based classification (gear alone decides).
- Depending on the externally-produced `drive-data.json` (we extract ourselves).

## Architecture

### Vendored official tool — `src/dcwb/vendor/tesla_dashcam/`

- `dashcam.proto` — schema, verbatim from the official repo.
- `dashcam_pb2.py` — generated with `grpcio-tools` and **committed** so runtime
  needs no protoc.
- `sei_extractor.py` — official SEI extraction logic, used as-is.
- `__init__.py`, `NOTICE` — NOTICE records origin (github.com/teslamotors/
  dashcam) and that reuse was confirmed acceptable by the project owner.

Dependencies: add `protobuf` to `[project.dependencies]`; add `grpcio-tools` to
the `dev` optional-dependencies (only needed to regenerate `dashcam_pb2.py`).

### New module — `src/dcwb/telemetry.py`

Thin wrapper over the vendored extractor.

```
@dataclass
class SegmentTelemetry:
    has_sei: bool
    frame_count: int
    gear_counts: dict[str, int]   # e.g. {"PARK": 147, "DRIVE": 2014}
    drove: bool                   # any DRIVE or REVERSE frame
    max_speed_mps: float          # captured for future use; not used in decisions

def read_segment_telemetry(front_clip: Path) -> SegmentTelemetry
```

- Calls the vendored `find_mdat` / `iter_sei_messages`.
- No SEI found → `SegmentTelemetry(has_sei=False, ...)`.
- What it does: turns one front clip into a gear summary. How to use it: call per
  segment. Depends on: vendored `tesla_dashcam` + `protobuf`.

### `prune.py` changes

In `find_candidates`, for each segment that already passed the age and
event-overlap guards, classify (when `use_telemetry` is true):

```
tel = read_segment_telemetry(front_clip)
if tel.has_sei and tel.drove:          -> protect (skip; real drive)
elif tel.has_sei and not tel.drove:    -> candidate, reason="parked-sei"
else (no SEI):                         -> motion score < threshold ? candidate, reason="low-motion" : skip
```

- `Candidate` gains `reason: str`.
- `segment_motion_score` / `frames_sampled` / `cameras_analyzed` remain for the
  fallback path.
- `quarantine` writes `reason` (and `gear_counts` when available) into each
  manifest row.
- When `use_telemetry` is false, behaviour is exactly today's pure-motion path.

### Config — `pipeline.json` `prune` section

- Add `use_telemetry: true` (default). Existing keys unchanged.

### Reporting — `format_report`

- Each segment line shows its `reason` (`parked-sei` / `low-motion`) so the SEI
  vs motion split is visible in the dry-run report.

## Data flow

`find_candidates` → per surviving segment → `read_segment_telemetry(front.mp4)`
→ branch → `Candidate(reason=...)` → `format_report` (dry-run) or `quarantine`
(manifest rows carry `reason`).

## Error handling

- Unreadable / corrupt clip or extractor exception → treat as `has_sei=False`
  (fall back to motion), never crash the run. Mirrors the existing
  `compute_motion_score` fail-safe (inf → never quarantine).
- Missing front clip in a segment → fall back to motion path.

## Testing (TDD)

1. `telemetry.py` — build a **synthetic SEI mp4** in-test: `ftyp` + `mdat`
   containing AVCC-framed SEI user_data_unregistered NALs whose payload is
   `42 42 42 69` + a `dashcam_pb2.SeiMetadata` serialized with known gears.
   Assert `read_segment_telemetry` returns correct `gear_counts` / `drove` /
   `has_sei` for: all-PARK, contains-DRIVE, and a no-SEI mp4.
2. `prune.py` — mock `read_segment_telemetry`; assert the three branches and the
   interaction with age/overlap guards; assert `use_telemetry=false` reproduces
   the pure-motion result (regression).
3. Existing prune tests stay green.

## Acceptance

- A driving segment with SEI is never quarantined regardless of pixel motion.
- A parked segment with SEI (all PARK) becomes a candidate (subject to guards).
- A no-SEI segment behaves exactly as today.
- `use_telemetry=false` is byte-for-byte the old behaviour.
