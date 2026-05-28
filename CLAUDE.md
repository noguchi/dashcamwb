# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`dcwb` は Tesla Model 3 Highland のドラレコ映像（6 カメラ）を D65 sRGB ニュートラルに補正する CLI／ローカル Web UI。設計仕様は `docs/superpowers/specs/2026-05-09-tesla-dashcam-wb-design.md`、README（日本語）に運用ワークフローがある。

## 開発コマンド

パッケージ管理・実行はすべて **uv** を使う（pip / venv は使わない）。`ffmpeg`/`ffprobe` が PATH 上に必須。Python 3.11+ は uv が自動調達する。

```bash
# セットアップ（uv.lock どおりに同期。dev 依存も含める）
uv sync --extra dev

# テスト（pyproject.toml で testpaths=tests, addopts=-v 固定）
uv run --extra dev pytest                                  # 全件
uv run --extra dev pytest tests/test_render.py             # 1 ファイル
uv run --extra dev pytest tests/test_render.py::test_name  # 1 ケース

# CLI 実行はすべて uv run 経由
uv run dcwb render <event_dir>
```

テストは合成 mp4 を **`libx264`** で生成・レンダーするため ffmpeg に libx264 が要る。実利用のデフォルトエンコーダは Apple Silicon の `h264_videotoolbox`（`--encoder` で切替）。

CLI サブコマンド（`pyproject.toml` の `[project.scripts]` で `dcwb` を公開）:
`calibrate` / `render <event_dir>` / `verify <event_dir>` / `render-all --source <dir>` / `serve` / `prune-recent`。

## アーキテクチャ

### 補正モデル: A×B の 2 レイヤを 1 つの 3×3 行列に合成

- **A レイヤ（カメラ固有キャスト）** — `calibrate.py` が過去クリップの昼間フレームからニュートラル候補ピクセルを統計マイニングし、カメラごとの対角ゲイン行列を `profiles/<camera>.json` に永続化。オフラインで一度だけ実行。
- **B レイヤ（シーン光）** — `render.py` の `estimate_scene_gain` がレンダー直前にクリップから数フレーム抜き、`awb.shades_of_gray`（Minkowski p=6）で照明色を推定。
- **合成** — `render.compose_clip_matrix` が `final = diag(scene_gain) @ profile.matrix_3x3` を作る。ここに 2 つの安全装置が同居している（**両方を一緒に読むこと**）:
  - いずれかの scene_gain チャネルが `[gain_min, gain_max]`（既定 0.7–1.5）外なら **B を完全破棄して A のみ**。
  - 夜間イベントは `attenuation` で B を恒等行列に向けて線形減衰（`night_attenuation`、既定 0.5）。

行列は **行ベクトル規約**で適用する: ピクセルは `flat @ matrix.T`（`render.py` / `verify.py` / `ffmpeg_wrap.render_with_matrix` の `colorchannelmixer` 引数生成で一貫）。ゲインは緑チャネル基準で `g_g ≡ 1` に正規化されている（`gain_min`/`gain_max` を 1 周りに対称配置するため）。

### データフロー

`/Volumes/sentryusb/{SentryClips,RecentClips,SavedClips}/`（読み取り専用）→ `render_event` → 既定で `/Users/noguchi/AI/dashcamwb/corrected/<event>/`（ハードコード、`--out-root` で変更可）。出力先には補正 mp4・`event.json` コピー・`thumb.png`・`_pipeline.json`（採用ゲインと最終行列のスナップショット）が並ぶ。元データはインプレース変更しない。

録画ファイルの実体は QNAP（ホスト名 `ts464.local`）に保存されており、Mac 側では `/Volumes/sentryusb/` にマウントして参照する。

### モジュール依存

`cli.py` がエントリ。`render.py` は `profile` / `matrix` / `awb` / `ffmpeg_wrap` / `daylight` / `calibrate`（タイムスタンプ・緯度経度の読取りヘルパ）に依存。`verify.py` は render と**同じ夜間 attenuation を適用**して 3 列（before / A only / A+B）の HTML を出す（ズレは回帰テスト対象）。`serve/` は Flask UI で、レンダーは最終的に `render_event` に集約される。

### serve（Flask UI）の要点

- `index.scan_sources` がイベント索引を作る。SentryClips/SavedClips は 1 サブディレクトリ = 1 イベント。**RecentClips はフラットな day-dir をファイル名タイムスタンプの 10 分ギャップで擬似イベントに自動グルーピング**（`RECENT_GAP_MINUTES`）。
- `render_jobs.JobQueue` は `max_workers=1` の直列キュー。`JobState` は全フィールド `_lock` 下で読み書きする。RecentClips の擬似イベントは、該当クリップだけを temp dir に **symlink** してから `render_event` に渡す（`glob("*.mp4")` がイベント範囲だけを拾うように）。
- 「レンダー済み」判定は「期待クリップ名の集合 ⊆ 既存 mp4 名」で行う（複数セグメントイベントの誤判定が過去のバグ。回帰テストあり）。
- `/corrected/...` ルートは `is_relative_to` でパストラバーサルを防ぐ。`serve` 起動時、CLI 側で全パス引数を `.resolve()` してから `create_app` に渡す（Flask の `send_file` は相対パスを CWD でなくパッケージディレクトリ基準で解決するため）。

### prune-recent（低モーション RecentClips の隔離）

`prune_mod.find_candidates` が `front` カメラの間引きフレーム間差分スコアで低モーションセグメントを検出し、`@dcwb_trash/` へ隔離（`manifest.jsonl` で管理）する。デフォルトはドライラン。`--apply` で隔離実行と期限切れ削除、`--purge` で削除のみ、`--restore SEGMENT_ID|all` で復元。`min_age_hours`（既定 48h）以内の新しいセグメントと、セグメントのタイムスタンプが SentryClips/SavedClips のイベント時間窓に入るものはスキップして保護する。パラメータは `pipeline.json` の `prune` セクション（`motion_threshold`, `frames_sampled`, `cameras_analyzed`, `min_age_hours`, `retention_days`, `trash_dir`）で調整可能。

### 時刻・昼夜判定

`daylight.is_daytime` は astral で日の出+30分〜日の入り−30分を昼と判定（tz-aware 必須、既定は Tokyo 緯度経度）。**Tesla の `event.json` timestamp や RecentClips のファイル名は naive なので JST(UTC+9) として解釈**する（`calibrate.JST`）。`calibrate` のサンプル予算は**イベント単位**（`max_per_event` をイベント内クリップに按分）。`event.json` の無い RecentScene day-dir では**クリップ単位のファイル名タイムスタンプ**で昼夜フィルタし、夜間クリップが profile を汚染しないようにする。

## 設定 `pipeline.json`

B レイヤのパラメータ（`awb.method`, `minkowski_p`, `samples_per_clip`, `saturation_high/low`, `gain_min/max`, `night_attenuation`）。`render`/`verify`/`serve` に `--pipeline-config` で差し替え可能。
