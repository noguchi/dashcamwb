# `dcwb serve` — ローカル UI 設計

USB マウントされた Tesla DashCam データを Web ブラウザで一覧・プレビューし、最新の WB キャリブレーションを実物に適用した結果を確認するためのローカルサーバ。

## 目的と非目標

**目的**

- `/Volumes/sentryusb` 配下の SentryClips / SavedClips / RecentClips をひとつの UI から横断ブラウズする。
- イベントごとに 6 カメラ × {before, after} の静止フレーム比較を高速に表示する。
- ボタン操作で `dcwb render` を起動し、補正済み動画をブラウザで再生する。

**非目標**

- マルチユーザ / 認証。シングルユーザのローカルツールに限定する。
- 動画編集・トリミング・SNS 共有等。
- USB 書き込み (USB は読み取り専用扱い)。

## アーキテクチャ

Flask ベースの同期サーバ。`render.py` / `verify.py` / `calibrate.py` は変更せず再利用する。新規モジュールは責務を分離する。

```
src/dcwb/serve/
  __init__.py
  app.py          # Flask app factory + ルート定義のみ
  index.py        # USB スキャン → Event 索引 (純関数)
  preview.py      # 静止フレーム before/after 生成 + ディスクキャッシュ
  render_jobs.py  # render_event ラッパー + ジョブ状態 (in-memory dict)
  templates/
    base.html.j2
    sources.html.j2
    events.html.j2
    event.html.j2
  static/
    app.css
```

`verify.py` の 1 フレーム抽出 + WB 適用ロジックは `preview.py` に共通化し、`verify.py` 側は薄いラッパーに更新する (重複コード排除)。**verify の出力 HTML と既存テストの挙動は変更しない** (リファクタは振る舞い保存)。

**新規依存**: `flask>=3.0` を `pyproject.toml` の `[project].dependencies` に追加。

## イベント索引 (`index.py`)

USB ルートから 3 種類のソースを統一スキーマでまとめる。

```python
@dataclass
class Event:
    source: str          # "SentryClips" | "SavedClips" | "RecentClips"
    name: str            # 表示用 ID
    path: Path           # 物理ディレクトリ (RecentClips では day-dir)
    clips: list[Path]    # 6 カメラ分の mp4 (バーチャル event なら範囲内のみ)
    start: datetime      # 最初のクリップ時刻
    end: datetime        # 最後のクリップ時刻 + 1分
    thumb: Path | None
```

**スキャン規則**

- **SentryClips / SavedClips**: 1 サブディレクトリ = 1 Event。`event.json` と `thumb.png` をそのまま利用。
- **RecentClips**: 各 `YYYY-MM-DD/` 配下の mp4 を時刻順に並べ、直前との時刻差が `RECENT_GAP_MINUTES = 10` 分超で区切って疑似 Event 化。`name` は先頭時刻 (例: `2026-05-08_0000`)、`thumb` は `None`。
- すべて `start desc` (新しい順) で返す。

**インタフェース**

```python
def scan_sources(usb_root: Path) -> dict[str, list[Event]]
```

- USB 未マウント時は空 dict を返す (例外を上げない)。
- メモリ計算量: 数百 Event × 数百 B → 全部メモリ展開で十分。
- 索引はサーバ起動時に 1 回構築。`POST /reindex` で明示再走査。自動再走査は無し。

## プレビュー生成 (`preview.py`)

イベント詳細ページ用に「中央 1 フレーム × 6 カメラ × {before, after}」の PNG を生成・キャッシュする。

```
cache/previews/<source>/<event_name>/
  <camera>_before.png
  <camera>_after.png
  meta.json
```

**ロジック**

- `extract_frame` で中央時刻のフレームを取得 → A 行列 (`prof.matrix_3x3`) で `before` を加工した `after_a`、`estimate_scene_gain` + `compose_clip_matrix` で `after_full` を生成 (verify.py と同じ B のかけ方)。プレビューでは `after_full` のみを `after` として保存する (1 枚に集約)。
- 解像度: 元 1280×960 をそのまま PNG。1 イベント ≈ 12 MB。
- 並列化: 6 カメラを `ThreadPoolExecutor(max_workers=6)` で並行処理。

**キャッシュ無効化**

`meta.json` に各 `profiles/<cam>.json` の mtime と `pipeline.json` の mtime を保存。ロード時に現在値と比較、不一致なら再生成。`meta.json` の形:

```json
{
  "profile_mtimes": {"front": 1715000000.0, "...": 0.0},
  "pipeline_mtime": 1715000001.0,
  "scene_gains": {"front": [1.18, 1.0, 1.14], "...": [1.0, 1.0, 1.0]},
  "errors": {"front": null, "back": "ffmpeg failed: ..."}
}
```

**インタフェース**

```python
@dataclass
class PreviewResult:
    paths: dict[str, dict[str, Path]]      # cam -> {"before": Path, "after": Path}
    scene_gains: dict[str, list[float]]    # cam -> [r, g, b]
    errors: dict[str, str | None]          # cam -> error message or None

def ensure_previews(
    event: Event,
    profiles_dir: Path,
    pipeline_cfg: dict,
    cache_root: Path,
) -> PreviewResult
```

テンプレートは `scene_gains` を添字表示、`errors` が非 None のセルにエラー文字列を出す。

**エラー処理**: clip 破損や ffmpeg 失敗時は該当カメラのみプレースホルダ PNG を出力し、`meta.json` にエラー文字列を残す。残り 5 カメラは描画続行。

## Render & 再生 (`render_jobs.py`)

「Render video」ボタン用の非同期ジョブと Range 対応の mp4 配信。

**ジョブモデル**

```python
@dataclass
class JobState:
    id: str
    source: str
    event_name: str
    status: Literal["queued", "running", "done", "failed"]
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
```

モジュールレベルの `dict[str, JobState]` で保持 (シングルプロセス Flask 前提)。

**キュー**

`concurrent.futures.ThreadPoolExecutor(max_workers=1)` でシリアル実行。VideoToolbox は GPU を 1 つ占有するため並列化メリット薄、USB 帯域競合も避ける。

**エントリポイント**

```python
def enqueue_render(event: Event, ...) -> str  # job_id
```

- `corrected/<event_name>/` に 6 つの mp4 が既に揃っていれば即 `done` ジョブを返し、render は走らせない。
- 既に同イベントの `running` ジョブがあれば既存 job_id を返す。

**render 実行**

既存 `dcwb.render.render_event` をそのまま呼ぶ。`out_root` は CLI と同じデフォルト (`/Users/noguchi/dashcamwb/corrected/`)。RecentClips の疑似 Event は範囲内 `clips` のみを `tempfile.TemporaryDirectory` 内に symlink して `render_event` に渡す。失敗は例外を `JobState.error` に文字列化して継続。

**進捗 API**

```
GET /jobs/<job_id> → {"status": ..., "error": ..., "elapsed_s": ...}
```

詳細ページは 2 秒間隔でポーリング、`done` になったらページリロード。

**動画配信**

```
GET /corrected/<source>/<event>/<cam>.mp4
```

専用ルートで `send_file(..., conditional=True)` を返し、HTTP Range をサポート → ブラウザ `<video>` でシーク可能。

## ルーティングと UI フロー

```
GET  /                           sources.html
GET  /s/<source>                 events.html
GET  /s/<source>/<event>/        event.html
GET  /preview/<source>/<event>/<cam>/<kind>.png   # kind ∈ {before, after}
POST /render/<source>/<event>    → 303 → 詳細ページ
GET  /jobs/<job_id>              JSON
GET  /corrected/<source>/<event>/<cam>.mp4   Range 対応
POST /reindex                    → 303 → トップ
```

**UI 構成**

- **トップ** (`sources.html`): 3 カード (SentryClips / SavedClips / RecentClips) に件数と概算サイズ。USB 未マウントなら警告バナー。
- **一覧** (`events.html`): 1 行 1 イベント。左に thumb (RecentClips は固定アイコン)、中央に時刻範囲とクリップ数、右に「rendered / rendering / not yet」バッジと「Open」リンク。新しい順。
- **詳細** (`event.html`): 6 カメラを 3×2 グリッド配置、各セルは横並び `before | after` の `<img>` + scene gain 添字。下部に「Render video」ボタン。`running` なら「Rendering… (経過時間)」、`done` なら 6 個の `<video controls>` がカメラ別に出る。`<script>` で `/jobs/<id>` を 2 秒ポーリング、`done` でページリロード。

**CSS**: `static/app.css` のみ。フレームワーク不使用。dark/light は `prefers-color-scheme` 追従。

## CLI 統合

```
dcwb serve [--source /Volumes/sentryusb]
           [--profiles-dir profiles]
           [--out-root /Users/noguchi/dashcamwb/corrected]
           [--pipeline-config pipeline.json]
           [--cache-dir cache]
           [--host 127.0.0.1]
           [--port 8765]
           [--debug]
```

`--source` のデフォルトは `/Volumes/sentryusb` (QNAP マウントポイント)。

## エラー処理

| ケース | 挙動 |
|---|---|
| USB 未マウント | `scan_sources` が空 dict、トップで黄色バナー + `/reindex` で再試行可。落ちない |
| `profiles/` 欠落 / カメラ不足 | 起動時にチェック → 赤バナー、詳細ページ該当セルは「profile missing」、ブラウズ自体は継続 |
| クリップ破損 / ffmpeg 失敗 | 該当カメラのみプレースホルダ PNG。render ジョブは `failed` + error 文字列をブラウザに表示 |
| 同イベント重複 render | 既存 `done` なら即返し、`running` なら既存 job_id を返却 |
| render 中に USB が外れる | ffmpeg がエラーで落ちて `failed` ジョブとして記録、サーバは継続 |

## テスト戦略 (`tests/test_serve.py`)

- **index**: 合成 USB ツリーを `tmp_path` に組み (SentryClips/`evt1`/6mp4、RecentClips/`2026-05-08`/12mp4 を時刻 0/15/30 分配置 → 3 疑似 event 期待)、`scan_sources` の戻り値を検証。
- **RecentClips グルーピング境界**: 10 分ぴったり / 10 分超 / 同時刻重複を表駆動で検証。
- **preview cache 無効化**: profile mtime 更新で再生成、未変更で再利用。`meta.json` の中身を検証。
- **app routes**: `app.test_client()` で 200/303 を確認。`enqueue_render` をモック化して即 `done` を返すフィクスチャを用意。
- **video range**: `Range: bytes=0-1023` で 206 が返り `Content-Range` が正しいことを確認。
- **CLI**: `dcwb serve --help` のパースのみ smoke test。Flask 起動自体は手動確認。

Flask は内蔵 `test_client` を使うので `pytest-flask` 不要。

## ディレクトリ追加

- `cache/` を `.gitignore` に追加 (プレビュー PNG は再生可能なアウトプット)。

## オープン質問

なし。実装プランへ進む。
