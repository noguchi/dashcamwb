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

テストは合成 mp4 を **`libx264`** で生成・レンダーするため ffmpeg に libx264 が要る。実利用のデフォルトエンコーダは Apple Silicon の `h264_videotoolbox`（`--encoder` で切替）。ただし `ffmpeg_wrap.resolve_encoder` が起動時に `ffmpeg -encoders` を調べ、要求エンコーダが無い環境（Linux など）では自動で `libx264` にフォールバックする（警告を stderr に出す）。`render`/`render-all`/`verify`/`highlight-day` すべてに効く。

CLI サブコマンド（`pyproject.toml` の `[project.scripts]` で `dcwb` を公開）:
`calibrate` / `render <event_dir>` / `verify <event_dir>` / `render-all --source <dir>` / `serve` / `prune-recent` / `highlight-day` / `sync-insta360` / `match-reference`。

`match-reference <参照動画> --recent <date> [--write]` は補正の **B レイヤ（シーン光）を「参照カメラ（Insta360/iPhone 等）とのマッチゲイン」に差し替える**ための参照ゲインを算出する。`sync.detect_visual_offset`（純画素フレーム差分の相互相関、IMU/GPS 不使用）で参照を Tesla front に時刻同期し、昼間・走行中のフレーム対から「A 適用後 Tesla」と「参照」の `awb.shades_of_gray` を取り、`refmatch.reference_gain`（`G = g_T/g_R`、緑正規化）をフレーム毎に求め**幾何平均**で集約。`.insv` は `ffmpeg_wrap.reframe_insv` で前方平面化、双方とも中央外景クロップしてから推定。結果 `(g_r,1,g_b)` を標準出力（人間可読＋JSON 1 行）、`--write` で `pipeline.json` の `awb.reference_gain` に書込む（他キー保持）。`reference_gain` 設定時は `render_event` が `estimate_scene_gain` を呼ばず全クリップで `scene_gain=reference_gain` を使う（未設定/`null` なら従来挙動）。`--max-window`（既定 600s、`compute_reference_gain(max_window=)`）が解析窓を参照先頭からこの長さに制限する（30 分級 `.insv` をネットワークマウント越しに全長リフレーム／全 front 連結しないため。CLI 側の front 選定窓と reframe duration・サンプリング上限の両方に効く）。目的は測色的正確さではなく**合成時の知覚的整合**（ADR 0001）。境界は `refmatch.py`、CLI は `cli.run_match_reference`。`sync-insta360 --reference-gain <R> <G> <B>`（または `--pipeline-config` の `awb.reference_gain`）で同じ色合わせを横並び合成の Tesla パネルに `render_sidebyside(right_matrix=diag(gain)@front_A)` として 1 パス焼き込める（明示指定が config より優先）。詳細は `docs/superpowers/specs/2026-05-31-reference-camera-match-gain-design.md`。

`highlight-day` は `RecentClips/<date>` の `front` カメラだけからドライブ記録向けハイライトを作る。テレメトリ（SEI の DRIVE/REVERSE）は「走った／停まった」だけを判定し、走行クリップの「ハイライトとしての良さ」は LAN 上の VLM が `interest`(0–10) で採点する（既定パス）。VLM 結果（`interest`/`scene_tags`/`caption`/`drive_quality`）は manifest にのみ記録し、字幕焼き込みはしない。VLM が到達不能なときは `--allow-no-ai` で MVP スコアラ（平均速度・速度変化・OpenCV の明るさ・画面変化量）にフォールバックする（指定なしならフレーム抽出前に中断）。`fast` と `cruise` の2スタイルがあり、出力 manifest には採用理由とスコア内訳・`skips` を必ず残す。出力は既定で `highlights/<date>/`（`--out-root`）に `highlight-<style>.mp4`・`highlight-<style>.json`・抜粋 `clips/`・`vlm-cache.json` が並ぶ。抜粋には render と同じ **A×B 白色補正行列**を適用し、さらに WB 後に creative「look」グレード（S字カーブ・彩度・ガンマ）を重ねて bt709 タグを焼く（Tesla のフラットな forensic 映像対策）。look は最終出力の concat フィルタ内で `setparams` により bt709 を確実にタグ付けする（出力 `-color_*` フラグだけでは効かない）。VLM 境界は `vlm.py`、選定/キャッシュは `highlight.py`、look は `ffmpeg_wrap.LookConfig`、設定は `pipeline.json` の `highlight_ai`／`look` セクション。詳細は `docs/superpowers/specs/2026-05-29-vlm-drive-highlight-design.md`。

`sync-insta360` は Insta360 の乗車視点 `.insv` と Tesla ドラレコ front クリップを時間同期し、(B) 横並び＋telemetry 焼き込み合成 mp4 と (C) `serve` の同期プレイヤー（手動ナッジ）を出す。同期は ① `.insv` の `creation_time`(UTC) と front ファイル名(JST) で**粗アンカー**（クリップ選定用。ファイル時刻のみ）→ ② **純粋に映像のフレーム差分（画面全体のエゴモーション）で「車が動き出す瞬間」を両映像で検出し相互相関**して精密化（`ffmpeg_wrap.frame_diff_envelope` で Insta 前レンズ生魚眼 `[0:v:0]` と Tesla front の差分エンベロープを取り、`sync.detect_visual_offset` で log1p→正規化相互相関。**加速度・ジャイロ・GPS・速度などのモーションデータは同期に一切使わない**）→ ③ プレイヤー上の手動ナッジで確定。**実機知見: creation_time アンカーはカメラ内蔵時計のズレで約20秒外れることがある（当データで visual_offset≈+19.5s, peak≈0.71 を自動補正）。旧来の Tesla `heading_deg`（当ファーム全ゼロ）/IMU ジャイロによる相関は弱く廃止。**`--visual-offset` で手動上書き、`--start-offset` で駐車区間をスキップして走行区間から描画。検出の信頼度 peak<0.30 なら粗アンカーにフォールバック。表示用乗車視点は `--insta-flat` の書き出し、無指定なら ffmpeg `v360` で dfisheye→平面に自動リフレーム（既定 `--insta-yaw 180`(前方)/`--insta-pitch 20`(操作パネルが入る下向き)/`--insta-roll 180`(上下補正)/`--insta-hfov 110`/`--insta-vfov 70`）。Insta360 IMU パーサ（`insta360.read_imu`、独自トレーラ・offsets-index・20バイト raw int16・1000Hz、トレーラ seek のみ）は残置するが現在 sync では未使用。出力は `--out-root`(既定 `sync-work`)/`sync/<date>/` に `combined-<date>.mp4`・`insta-flat.mp4`・`tesla-concat.mp4`・`sync.json`(δ=プレイヤーの右トリム量・信頼度・telemetry・パス)・`telemetry.ass`。モジュール: `insta360.py`(creation_time＋トレーラ IMU パーサ、vendored `vendor/insta360/NOTICE`)、`sync.py`(`resample_uniform`/`normalized_xcorr`/`detect_visual_offset`/`select_front_clips`/`telemetry_ass`/`write_sync_manifest`)、`telemetry.iter_segment_frames`(per-frame SEI、オーバーレイ表示用)、`ffmpeg_wrap.frame_diff_envelope`/`render_sidebyside`/`reframe_insv`、`serve` の `/sync/<date>` プレイヤー＋`/sync-nudge`。詳細は `docs/superpowers/specs/2026-05-30-insta360-tesla-sync-design.md`。

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

判定は既定で **gear 主・モーション補助**: `telemetry.read_segment_telemetry` が front クリップの埋め込み SEI（Tesla 公式 `dashcam.proto`、vendored `src/dcwb/vendor/tesla_dashcam/`）から `gear_state` を読み、DRIVE/REVERSE を含めば走行として保護、全 PARK なら候補（reason=`parked-sei`）。SEI 無しは従来のモーションスコアにフォールバック（reason=`low-motion`）。`prune.use_telemetry=false` で純モーションに戻る。SEI は firmware 2025.44.25+/HW3+ かつ駐車中は欠落しうるため、フォールバックは必須。

### 時刻・昼夜判定

`daylight.is_daytime` は astral で日の出+30分〜日の入り−30分を昼と判定（tz-aware 必須、既定は Tokyo 緯度経度）。**Tesla の `event.json` timestamp や RecentClips のファイル名は naive なので JST(UTC+9) として解釈**する（`calibrate.JST`）。`calibrate` のサンプル予算は**イベント単位**（`max_per_event` をイベント内クリップに按分）。`event.json` の無い RecentScene day-dir では**クリップ単位のファイル名タイムスタンプ**で昼夜フィルタし、夜間クリップが profile を汚染しないようにする。

## 設定 `pipeline.json`

B レイヤのパラメータ（`awb.method`, `minkowski_p`, `samples_per_clip`, `saturation_high/low`, `gain_min/max`, `night_attenuation`）。`render`/`verify`/`serve` に `--pipeline-config` で差し替え可能。`prune` セクション（prune-recent 用）、`highlight_ai` セクション（VLM エンドポイント・サンプリングパラメータ）、`look` セクション（highlight の creative グレード: `scurve`/`saturation`/`gamma`/`tag_bt709`）も同じファイルに同居する。
