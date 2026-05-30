# dcwb — Tesla DashCam White Balance

Tesla Model 3 Highland のドライブレコーダ映像（SentryClips / SavedClips / RecentClips）を、D65 sRGB の白に合わせて補正する CLI／ローカル Web UI です。

設計仕様: [`docs/superpowers/specs/2026-05-09-tesla-dashcam-wb-design.md`](docs/superpowers/specs/2026-05-09-tesla-dashcam-wb-design.md)

## このツールが解決する課題

Tesla 純正の DashCam 出力は、6 つのカメラ（front / back / 左右ピラー / 左右リピータ）でそれぞれホワイトバランスが微妙にズレています。同じ白い対象物でも front は赤め、left_pillar は青め、といった具合です。素のままで動画を並べると違和感が出ますし、後で映像を二次利用するときに色味が揃わなくて困ります。

dcwb は次の 2 段構えで補正します。

1. **A: カメラごとの恒常的なキャスト**を、過去のクリップから統計的に推定して 3x3 行列にする（オフラインで一度だけ）。
2. **B: クリップ単位のシーン光（昼/夕/夜の色温度差）**を Shades-of-Gray 法でその場で推定し、A と合成して最終 3x3 行列にする。

夜間・薄暮のイベントでは B の影響が過大になりがちなので、`night_attenuation` で減衰させた行列を使います。

## セットアップ

パッケージ管理・実行はすべて [uv](https://docs.astral.sh/uv/) を使います。`ffmpeg` が PATH に必要です（Apple Silicon では実利用時 `h264_videotoolbox`、テストでは `libx264` を使用）。Python 3.11+ は uv が自動で用意します。

```bash
# 実行環境を作成（uv.lock どおりに同期。dev 依存も含める）
uv sync --extra dev

# 以後はすべて uv run 経由で実行（毎回 uv.lock に沿って自動同期してから呼ぶ）
uv run dcwb --help
```

## サブコマンド一覧

`dcwb` は 9 個のサブコマンドを公開します。役割で 4 グループに分けると見通しが良いです。補正の核は **A×B の 2 レイヤを 1 つの 3×3 行列に合成**する設計で（A=カメラ固有キャスト／B=シーン光）、各コマンドはこのモデルのどこかに関わります。

| サブコマンド | 役割 | 主な引数 |
|------|------|------|
| [`calibrate`](#1-キャリブレーション初回のみカメラ交換やファーム更新時に再実行) | **A レイヤ**を作る（初回/カメラ交換時のみ） | `--source` `--profiles-dir` `--max-samples-per-event` |
| [`render`](#3-映像の補正cli-単発一括) | 1 イベント（6 カメラ）を A×B で補正 | `event_dir` `--out-root` `--pipeline-config` `--encoder` |
| [`render-all`](#3-映像の補正cli-単発一括) | ディレクトリ内の全イベントを一括補正 | `--source` `--out-root` `--pipeline-config` |
| [`verify`](#2-キャリブレーションの妥当性確認) | before / A only / A+B の 3 列 HTML を生成（QC） | `event_dir` `--out-html` `--pipeline-config` |
| [`serve`](#4-ブラウザ-uiインタラクティブ運用) | ローカル Web UI（索引→レンダー・同期プレイヤー） | `--source` `--out-root` `--host` `--port` |
| [`prune-recent`](#recentclips-の整理低モーションクリップの隔離) | 低モーション RecentClips を隔離（容量管理） | `--apply` `--purge` `--restore` `--retention-days` |
| [`highlight-day`](#ドライブハイライト日別) | 日別ドライブ・ハイライトを生成（front のみ＋VLM 採点） | `--date` `--style` `--allow-no-ai` `--no-look` |
| [`sync-insta360`](#insta360-との同期sync-insta360) | Insta360 乗車視点と Tesla front を時刻同期・横並び合成 | `insv...` `--recent` `--reference-gain` `--visual-offset` |
| [`match-reference`](#参照カメラへの色合わせmatch-reference) | 参照カメラへの色合わせゲイン（B レイヤ差し替え）を算出 | `reference` `--recent` `--max-window` `--write` |

### 補正の核（A×B モデル）

- **`calibrate`** — 過去クリップの**昼間フレーム**からニュートラル候補ピクセルを統計マイニングし、カメラごとの対角ゲイン行列（A）を `profiles/<camera>.json` に永続化。オフラインで一度だけ。以降の全 render の土台。
- **`render`** — 1 イベントに **A×B 行列**を適用して補正 mp4 を出力。B は既定で per-clip の gray-world 推定。`awb.reference_gain` 設定時は B を**参照ゲイン**に差し替え。出力先に `_pipeline.json`（採用ゲイン・最終行列のスナップショット）も残す。元データは不変。
- **`render-all`** — `render` をディレクトリ配下の全イベントに回すバッチ版。1 イベント失敗しても継続。
- **`verify`** — **before / A only / A+B** の 3 列 HTML を生成。render と同じ夜間 attenuation を適用するので、補正の効きをブラウザで比較できる（ファイルは書き換えない）。

### インタラクティブ運用

- **`serve`** — Flask UI でイベント索引→レンダーを操作（直列キュー `max_workers=1`）。**RecentClips はファイル名タイムスタンプの 10 分ギャップで擬似イベントに自動グルーピング**。`/sync/<date>` の同期プレイヤー（手動ナッジ）もここ。

### 運用補助

- **`prune-recent`** — front の埋め込み SEI テレメトリ（`gear_state`）で**走行=保護／全 PARK=候補**を判定（SEI 無しはモーションスコアにフォールバック）。`@dcwb_trash/` へ隔離（既定ドライラン、`--apply` で実行、`--restore` で復元）。新しいセグメントとイベント時間窓に入るものは保護。
- **`highlight-day`** — `RecentClips/<date>` の **front だけ**からハイライトを作成。走行判定は SEI、「ハイライトとしての良さ」は LAN 上の **VLM が interest(0–10)** で採点（到達不能時は `--allow-no-ai` で MVP スコアラ）。抜粋には render と同じ **A×B 補正** ＋ creative「look」グレード（S 字・彩度・bt709 タグ）を重ねる。

### 一般カメラとの合成（色合わせ）

- **`match-reference`** — 参照（Insta360/iPhone 等）を Tesla front に映像差分で同期し、昼間・走行フレーム対から **B レイヤ差し替え用の参照ゲイン `(g_r,1,g_b)`** を算出。`--write` で `pipeline.json` の `awb.reference_gain` に書込み、以降の `render` / `sync-insta360` がそれを使う。目的は測色精度ではなく**合成時の知覚的整合**。
- **`sync-insta360`** — `.insv` と Tesla front を**純粋な映像フレーム差分の相互相関**で同期（IMU/GPS 不使用）し、横並び合成 mp4 ＋ serve の同期プレイヤーを出力。`--reference-gain`（または pipeline.json の `awb.reference_gain`）で **色合わせを Tesla 側パネルに焼き込む**。

### 典型ワークフロー

```
calibrate(初回) → serve で日々確認/補正 ┐
                                        ├ render / render-all で一括出力
prune-recent で容量管理 ────────────────┘
highlight-day で日別ハイライト
[一般カメラと合成する場合] match-reference --write → sync-insta360 --reference-gain
```

## 4 ステップの基本ワークフロー

### 1. キャリブレーション（初回のみ／カメラ交換やファーム更新時に再実行）

```bash
uv run dcwb calibrate \
  --source /Volumes/sentryusb \
  --profiles-dir profiles \
  --max-samples-per-event 3
```

`--source` 配下の `SentryClips/`, `RecentClips/`, `SavedClips/` をすべて走査し、**昼間の**フレームから「ニュートラル候補ピクセル」を集めて、6 カメラ分の `profiles/<camera>.json` を出力します。

- `--max-samples-per-event` は **イベント単位の予算**です。1 イベント内のクリップ数で按分されるので、長時間 RecentClips が結果を独占しません。
- RecentClips のように `event.json` がない day-dir では、ファイル名のタイムスタンプ（JST と解釈）で昼夜判定するので、夜間クリップが混入しません。

### 2. キャリブレーションの妥当性確認

```bash
uv run dcwb verify /Volumes/sentryusb/SentryClips/2026-05-05_13-50-46 \
  --out-html verify.html
open verify.html
```

カメラごとに「補正前」「A のみ」「A+B（最終）」の 3 列が並びます。白い対象物がニュートラルに見え、6 カメラで色味が揃っているか目視で確認してください。夜間イベントでは `night_attenuation` 適用後の見た目になります。

### 3. 映像の補正（CLI 単発／一括）

単一イベント:

```bash
uv run dcwb render /Volumes/sentryusb/SentryClips/2026-05-05_13-50-46
# → corrected/2026-05-05_13-50-46/ に出力
```

ディレクトリ配下を一括:

```bash
uv run dcwb render-all --source /Volumes/sentryusb/SentryClips
```

出力先は `--out-root` で変更可。各イベントディレクトリには元のクリップと同名の補正済み mp4、`event.json` のコピー、補正パラメータを記録した `_pipeline.json` が並びます。

### 4. ブラウザ UI（インタラクティブ運用）

```bash
uv run dcwb serve --source /Volumes/sentryusb
```

<http://127.0.0.1:8765/> を開くと、

- SentryClips / SavedClips / RecentClips の一覧（RecentClips は 10 分ギャップでイベントに自動グルーピング）
- 各イベントのプレビュー画像（カメラごとに before/after 1 枚ずつ）
- 「Render video」ボタンで該当イベントを補正してその場で再生（複数セグメントのイベントもクリップごとに `<video>` が並びます）

を扱えます。背後でジョブキューが動き、補正済み mp4 は `corrected/<event>/` に蓄積されます。

## RecentClips の整理（低モーション・クリップの隔離）

動きの少ない RecentClips を自動で隔離し、保持期間経過後に削除します。SentryClips / SavedClips には一切触れません。

```bash
# まずドライラン（何も消さない、候補をレポート表示）
uv run dcwb prune-recent --source /Volumes/sentryusb

# 問題なければ隔離実行（@dcwb_trash へ移動、期限切れは同時に削除）
uv run dcwb prune-recent --source /Volumes/sentryusb --apply

# 誤って隔離したセグメントを元に戻す
uv run dcwb prune-recent --source /Volumes/sentryusb --restore 2026-05-08_00-00-00
# すべて戻す場合は all
uv run dcwb prune-recent --source /Volumes/sentryusb --restore all
```

- 直近 48 時間のクリップ、および SentryClips/SavedClips のイベント時間に重なるクリップは保護されます。
- 隔離されたファイルは `@dcwb_trash/` に 14 日間保持され、`--apply` または `--purge` 実行時に期限切れ分が削除されます。
- `--purge` 単体で、期限切れ trash の削除だけを実行することもできます（`--apply` 時は隔離後に自動で実行）。
- 閾値・保持期間は `pipeline.json` の `prune` セクションで調整できます。
- **走行判定**: 既定（`use_telemetry: true`）では、Tesla が各 mp4 に埋め込む SEI テレメトリの `gear_state` を読み、DRIVE/REVERSE を含むセグメントは「走行」として保護します。SEI が無いセグメント（駐車中や旧 firmware）は従来どおり front カメラのモーションスコアで判定します。`pipeline.json` の `prune.use_telemetry` を `false` にすると純モーション動作に戻ります。

## ドライブ・ハイライト（日別）

`RecentClips/<date>` から front カメラのみのハイライト動画を作れます。危険検出ではなく、見返して楽しいドライブ記録向けです。

```bash
# テンポ重視: 短い切り出しを多めにつなぐ
uv run dcwb highlight-day --source /Volumes/sentryusb --date 2026-05-08 --style fast

# ドライブ感重視: 長めの区間を少なめにつなぐ
uv run dcwb highlight-day --source /Volumes/sentryusb --date 2026-05-08 --style cruise
```

出力は `highlights/<date>/highlight-<style>.mp4` と `highlight-<style>.json`、抜粋クリップ `clips/`、`vlm-cache.json` です。Tesla SEI テレメトリで DRIVE/REVERSE が確認できた front クリップだけを対象にします。SEI が無いクリップも含めたい場合は `--allow-no-sei` を指定できますが、manifest では低信頼として記録されます。

抜粋クリップには `render` と同じ **A×B 白色補正行列**を適用し、さらに白色補正の後段で creative な「ルック」グレード（緩い S カーブ＋彩度/ガンマ）を重ねます。Tesla のフラットで低彩度な forensic 映像に締まりを出すためで、最終出力には bt709 のカラーメタデータをタグ付けします（`--no-look` でルックを無効化）。

### 選定スコア（AI 既定 / MVP フォールバック）

既定では、走行と判定された各クリップを LAN 上の Vision-Language Model（VLM）で採点します。テレメトリは「走った／停まった」だけを判定し、「ハイライトとしてどれだけ良いか」は VLM が `interest`（0–10）で評価します。各クリップから 3 フレーム（10% / 50% / 90%）を抜き出して 1 回の呼び出しにまとめ、結果（`interest` / `scene_tags` / `caption` / `drive_quality`）は manifest にのみ記録します（字幕焼き込みはしません）。VLM の結果は `highlights/<date>/vlm-cache.json` にクリップ単位でキャッシュされ、`fast` と `cruise` は同じ 1 日分のキャッシュを共有します。

VLM サーバは OpenAI 互換エンドポイント（例: LM Studio）で、`pipeline.json` の `highlight_ai` セクションで設定します。

```bash
# 設定を上書きして実行
uv run dcwb highlight-day --source /Volumes/sentryusb --date 2026-05-08 --style cruise \
  --vlm-endpoint http://galleria.local:1234/v1 --vlm-model google/gemma-4-26b-a4b
```

- `--vlm-endpoint` / `--vlm-model`: `highlight_ai.endpoint` / `.model` を上書き。
- `--allow-no-ai`: VLM が到達不能なとき、エラーで止めずに従来の MVP スコアラ（速度・速度変化・画面変化・明るさ）にフォールバックします（manifest の `selection` は `mvp-fallback`）。指定しない場合、エンドポイントに到達できなければフレーム抽出や ffmpeg を実行する前に中断します（無駄な処理をしない）。
- `--no-vlm-cache`: その日のキャッシュを無視して再取得します。

VLM はビジョン対応モデルが必要です（Gemma なら 3 系以降のマルチモーダル、本リポジトリの既定は `google/gemma-4-26b-a4b`）。LM Studio が `response_format: json_schema` を honor しない場合は `highlight_ai.use_json_schema` を `false` にすると、プロンプト誘導 + 寛容なパースに切り替わります。

## Insta360 との同期（`sync-insta360`）

乗車視点の Insta360 映像（`.insv`）と Tesla ドラレコ front クリップを時間同期し、横並び合成 mp4 と Web プレイヤーを生成します。

設計仕様: [`docs/superpowers/specs/2026-05-30-insta360-tesla-sync-design.md`](docs/superpowers/specs/2026-05-30-insta360-tesla-sync-design.md)

```bash
uv run dcwb sync-insta360 <event.insv ...> --recent <YYYY-MM-DD> [--insta-flat <flat.mp4>] [--out-root sync-work]
# then: uv run dcwb serve --out-root sync-work  → open /sync/<YYYY-MM-DD> to fine-tune with the nudge slider
```

同期は 3 段階のハイブリッド方式で行います。まず `.insv` の `creation_time`(UTC) と front ファイル名(JST) で粗オフセットを算出し（①タイムスタンプ・アンカー）、次に Tesla の GPS course 由来ヨーレートと Insta360 IMU ジャイロの相互相関で精密化し（②クロスコリレーション）、最後に Web プレイヤーの手動ナッジで確定します（③）。相互相関の信頼度が低い場合（< 0.35）は①アンカーへフォールバックします。

出力は `--out-root`（既定 `sync-work`）の `sync/<YYYY-MM-DD>/` に `combined-<date>.mp4`（横並び＋テレメトリ字幕）、`sync.json`（δ・信頼度・telemetry・パス）が並びます。`--insta-flat` を省略すると ffmpeg の `v360` フィルタで dfisheye→平面に自動リフレームします。

`--reference-gain <R> <G> <B>` を渡すと（または `--pipeline-config` の `awb.reference_gain` が設定されていれば）、Tesla 側パネルに `diag(reference_gain) @ front_A` を 1 パスで焼き込み、合成を乗車視点の発色に揃えます（`match-reference` の出力。明示指定が pipeline.json より優先）。未指定なら従来どおり生の front をそのまま並べます。Insta360 の 18GB 本体ファイルは末尾のトレーラ（IMU インデックス）だけを `seek` して読むため、フル転送は不要です。

## 参照カメラへの色合わせ（`match-reference`）

一般カメラ（Insta360 / iPhone 等）の映像と Tesla 6 カメラ映像を合成しても違和感が出ないよう、補正の **B レイヤ（シーン光）を「参照カメラとのマッチゲイン」に差し替える**機能です。参照を Tesla front に映像差分で時刻同期し、昼間・走行中のフレーム対から「A 適用後 Tesla」と「参照」の Shades-of-Gray を取り、その比の幾何平均を参照ゲイン `(g_r, 1, g_b)` として算出します。目的は測色的正確さ（true D65）ではなく**合成時の知覚的整合**です（参照が較正済み基準でなくてよい）。

設計仕様: [`docs/superpowers/specs/2026-05-31-reference-camera-match-gain-design.md`](docs/superpowers/specs/2026-05-31-reference-camera-match-gain-design.md) / [`docs/adr/0001-consumer-camera-reference-color-matching.md`](docs/adr/0001-consumer-camera-reference-color-matching.md)

```bash
# 算出して標準出力（人間可読 + JSON 1 行）。--write で pipeline.json に書き込む
uv run dcwb match-reference <reference.insv|flat.mp4> --recent <YYYY-MM-DD> [--samples 10] [--max-window 600] [--write]
# その後 render/render-all すると 6 カメラ・全クリップが参照トーンに揃う
uv run dcwb render <event_dir> --pipeline-config pipeline.<YYYY-MM-DD>.json
# Insta360 との横並び合成にも同じ色合わせを焼き込める（下記）
uv run dcwb sync-insta360 <reference.insv> --recent <YYYY-MM-DD> --reference-gain <R> <G> <B>
```

算出した `awb.reference_gain` は**シーン／光源依存**の値です。汎用デフォルトではなく、**合成対象のドライブごとの `pipeline.json` に置く**運用を想定しています（`match-reference --write` で生成）。`reference_gain` が未設定（`null`）なら **従来どおり**の per-clip gray-world 推定にフォールバックし、既存挙動は一切変わりません。参照ゲインが妥当域（`gain_min`/`gain_max`）外なら render の既存安全網で B を破棄して A のみになります。

`--max-window`（既定 600 秒）は解析窓を参照先頭からこの長さに制限します。30 分級の `.insv` をネットワークマウント越しに**全長リフレーム／全 front 連結すると現実的でない**ため、先頭の数百秒（昼間・走行中なら十分安定）だけで算出します。

## 設定 (`pipeline.json`)

リポジトリ同梱のデフォルトは以下です。CLI の `--pipeline-config` で差し替え可能。

```json
{
  "awb": {
    "method": "shades_of_gray",
    "minkowski_p": 6,
    "samples_per_clip": 10,
    "saturation_high": 0.97,
    "saturation_low": 0.03,
    "gain_min": 0.7,
    "gain_max": 1.5,
    "night_attenuation": 0.5,
    "reference_gain": null
  },
  "highlight_ai": {
    "endpoint": "http://galleria.local:1234/v1",
    "model": "google/gemma-4-26b-a4b",
    "api_key": "lm-studio",
    "frames_per_clip": 3,
    "frame_max_edge": 512,
    "timeout_sec": 120,
    "max_retries": 1,
    "interest_min": 1,
    "system_prompt": null,
    "use_json_schema": true,
    "max_tokens": 768,
    "temperature": 0.2,
    "repeat_penalty": 1.4,
    "frequency_penalty": 0.5
  },
  "look": {
    "scurve": "0/0 0.25/0.21 0.5/0.5 0.75/0.82 1/1",
    "saturation": 1.12,
    "gamma": 1.03,
    "tag_bt709": true
  }
}
```

| キー | 意味 |
|------|------|
| `samples_per_clip` | 1 クリップから抜くフレーム枚数（B レイヤ推定用） |
| `saturation_high` / `saturation_low` | Shades-of-Gray で除外する飽和上下限 |
| `gain_min` / `gain_max` | B レイヤのゲインがこの範囲外なら B を破棄して A のみで補正 |
| `night_attenuation` | 夜間判定時に B レイヤを 1.0 に向けて線形減衰させる係数 |
| `reference_gain` | `[g_r, g_g, g_b]` を設定すると B レイヤを参照カメラのマッチゲインに差し替え（6 カメラ・全クリップ一律）。`null` で従来の per-clip 推定。`match-reference --write` で生成 |
| `highlight_ai.endpoint` / `.model` | ハイライト選定に使う VLM の OpenAI 互換エンドポイントとモデル ID |
| `highlight_ai.frames_per_clip` / `.frame_max_edge` | 1 クリップから VLM に送るフレーム数と長辺リサイズ上限(px) |
| `highlight_ai.interest_min` | この `interest` 未満のクリップは選定から除外 |
| `highlight_ai.system_prompt` | 評価基準のシステムプロンプト（`null` で組み込みデフォルト） |
| `highlight_ai.use_json_schema` | `response_format: json_schema` を使う。`false` でプロンプト誘導パースに切替 |
| `highlight_ai.max_tokens` | VLM の最大生成トークン。小さすぎると生成が打ち切られ繰り返しループに陥る（既定 768） |
| `highlight_ai.temperature` / `.repeat_penalty` / `.frequency_penalty` | サンプリング。LM Studio の UI 設定は OpenAI 互換 API に効かないため、ここから毎リクエスト送る。`repeat_penalty`/`frequency_penalty` は繰り返しループの抑制に効く |
| `look.scurve` / `.saturation` / `.gamma` | ハイライト excerpt に WB の後段で乗せる“ルック”グレード。Tesla のフラット・低彩度な絵に緩い S カーブ＋彩度/ガンマで締まりを出す。`--no-look` で無効化 |
| `look.tag_bt709` | 出力に bt709 のカラーメタデータを付与（Tesla の無タグ stream をプレイヤーが誤解釈しないように） |

## テスト

```bash
uv run --extra dev pytest -q
```

合成 mp4（`libx264`）でレンダリングまで含めて検証します。回帰テストとして以下を含みます:

- 複数セグメントイベントの「レンダー済み」誤判定
- `calibrate` のイベント単位サンプリングと RecentClips 夜間フィルタ
- `/corrected/...` ルートのパストラバーサル防止
- preview / verify が render と同じ夜間 attenuation を適用すること

## ディレクトリ構成

```
src/dcwb/
  cli.py            # サブコマンドのエントリポイント
  calibrate.py      # A レイヤ（恒常キャスト）の統計マイニング
  render.py         # A+B 合成 + ffmpeg 呼び出し
  verify.py         # before / A only / full の HTML レポート
  awb.py            # Shades-of-Gray 実装
  matrix.py         # 3x3 行列ヘルパ
  profile.py        # Profile / CalibrationMeta データクラス
  daylight.py       # astral で昼夜判定
  ffmpeg_wrap.py    # ffmpeg / OpenCV 抽出ラッパ + LookConfig（ルックグレード）
  highlight.py      # ドライブ・ハイライト選定 + VLM キャッシュ
  vlm.py            # VLM クライアント境界（OpenAI 互換 / 構造化出力）
  telemetry.py      # Tesla SEI（gear_state 等）の読取り
  prune.py          # 低モーション RecentClips の隔離
  vendor/tesla_dashcam/  # Tesla 公式 dashcam.proto（vendored）
  serve/            # Flask UI（app.py / preview.py / render_jobs.py / index.py）
  templates/        # verify.html.j2
tests/              # pytest（synthetic mp4 fixtures）
docs/superpowers/   # 設計仕様 / 実装計画
```
