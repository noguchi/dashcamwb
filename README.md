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
    "night_attenuation": 0.5
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
