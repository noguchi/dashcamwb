# RecentClips 低モーション・クリップ自動隔離 設計仕様

- **作成日**: 2026-05-28
- **対象**: QNAP に蓄積された `RecentClips` のうち、動きの少ない（利用価値の低い）クリップを安全に隔離・削除する仕組み
- **位置づけ**: 設計仕様（design spec）。実装計画は別途 `writing-plans` で作成する。

---

## 1. 目的とスコープ

### 1.1 達成したいこと
QNAP（`ts464.local`、Mac では `/Volumes/sentryusb/` にマウント）に溜まり続ける `RecentClips` のうち、**動きがほとんど無い低価値クリップ**を自動で見分けて隔離し、保持期間経過後に削除する CLI を追加する。自宅ガレージ駐車中の映像が典型例だが、判定は場所ではなく**モーション（動きの少なさ）**で行う。

### 1.2 安全前提（ブレインストーミングで確認済み）
- **Sentry Mode は自宅でも常時 ON**。ガレージで Sentry のトリガー閾値を超える出来事が起きれば、必ず `SentryClips` に**独立した別録画**が生成される（Tesla は RecentClips を SentryClips に「移動」するのではなく別録画する）。したがって RecentClips を消しても、トリガー済みイベントの記録は SentryClips 側に無傷で残る。
- ただし「走行中・低モーション」（渋滞・信号待ち等）の最中の接触事故は、走行中ゆえ Sentry が作動せず、低モーションの RecentClips が唯一の記録になり得る。この取りこぼしを設計で守る（→ 3章ガード ＋ 隔離方式）。

### 1.3 スコープ
- **対象**: `RecentClips` のみ。
- **非対象**: `SentryClips` / `SavedClips` には一切触れない（読みもするがファイルは変更・削除しない）。

### 1.4 入力データの実態
- `RecentClips/<YYYY-MM-DD>/` の **フラットな day-dir** に、`<YYYY-MM-DD_HH-MM-SS>-<camera>.mp4` 形式のクリップが並ぶ。
- **event.json は無い**ため GPS（`est_lat`/`est_lon`）も `reason` も付かない。利用できるメタデータは**ファイル名のタイムスタンプのみ**。これが「位置ではなくモーションで判定する」根拠。
- SMB マウントは**書き込み可能**（実測確認済み）。QNAP の `@Recycle` / `@Recently-Snapshot` はほぼ非活性で復旧網として当てにできないため、**dcwb 自前で trash を管理**する。

---

## 2. アーキテクチャ全体像

```
┌──────────────────────────────────────────────────────────────┐
│ /Volumes/sentryusb (QNAP, SMB 書き込み可)                      │
│  - RecentClips/<date>/<ts>-<cam>.mp4   ← 判定・隔離の対象      │
│  - SentryClips/<event>/event.json      ← overlap guard 参照のみ│
│  - SavedClips/<event>/event.json       ← overlap guard 参照のみ│
│  - @dcwb_trash/                         ← 隔離先（自前 trash）  │
└───────────────┬──────────────────────────────────────────────┘
                │
   user CLI     │
   $ dcwb prune-recent              （既定: ドライラン）
   $ dcwb prune-recent --apply      （隔離実行）
   $ dcwb prune-recent --purge      （保持期間超過を実削除）
   $ dcwb prune-recent --restore ID （元の場所へ復旧）
                │
   ┌────────────▼─────────────┐      ┌──────────────────────────┐
   │ prune.py                 │      │ index.py（流用）          │
   │  - segment 化            │◀────▶│  _group_recent_day        │
   │  - motion score 計算     │      │  scan_sources(Sentry/Saved)│
   │  - 候補選定 / ガード      │      └──────────────────────────┘
   │  - 隔離 / purge / restore│
   │  - manifest.jsonl 管理   │      ┌──────────────────────────┐
   └────────────┬─────────────┘      │ ffmpeg_wrap.py（拡張）     │
                │                     │  複数フレーム1パス抽出     │
                └────────────────────▶│  （新ヘルパ）              │
                                      └──────────────────────────┘
```

依存方針: 既存の `index._group_recent_day`（擬似イベント分割）と `calibrate._parse_clip_ts`（ファイル名→JST）を再利用。フレーム抽出は `ffmpeg_wrap` を拡張。設定は `pipeline.json` を踏襲。

---

## 3. コンポーネント設計

### 3.1 CLI サブコマンド `dcwb prune-recent`
`cli.py` にサブコマンドを追加。

| モード | フラグ | 動作 |
| --- | --- | --- |
| ドライラン（既定） | （なし） | 隔離候補をレポート表示するだけ。ファイルは一切変更しない。 |
| 隔離 | `--apply` | 候補を `@dcwb_trash/` へ move。manifest に追記。実行時に purge も走らせる。 |
| 実削除 | `--purge` | manifest 上で保持期間超過の隔離物を実削除し manifest を更新。 |
| 復旧 | `--restore <id\|all>` | manifest を参照して元の相対パスへ move し戻す。 |

共通オプション: `--source <usb_root>`（既定 `/Volumes/sentryusb`）、`--pipeline-config <path>`、`--retention-days N`（manifest 設定を上書き）。

### 3.2 モーション判定（新モジュール `prune.py`）
- **判定単位 = タイムスタンプ・セグメント**。`<ts>-<cam>.mp4` の同一 `<ts>` を持つ6カメラを1セグメント（≒1分）として扱う。
- **手順**:
  1. `cameras_analyzed`（既定 `["front"]`）のクリップから、1回の ffmpeg パスで `frames_sampled`（既定 8）枚を等間隔抽出。
  2. 各フレームをグレースケール＋小サイズ（例 64×64）に縮小。
  3. 隣接フレーム間の平均絶対差を計算し、その**最大値**を motion score とする。
  4. `motion score < motion_threshold` なら「低モーション（静止）」と判定。
- 低モーションと判定したセグメントは、**同一 `<ts>` の全6カメラ**を隔離対象にする（front だけ消すと再生時に欠けるため）。
- レポート表示は `index._group_recent_day` の擬似イベント単位でまとめ、各セグメントの motion score・サイズ・隔離可否（ガード理由含む）を出す。

### 3.3 安全ガード
1. **min-age ガード**: セグメント時刻が現在より `min_age_hours`（既定 48h）以内なら対象外。ライブ・ローリングバッファや直近映像を守る。
2. **overlap ガード**: `scan_sources` で得た `SentryClips` / `SavedClips` の各イベントの時間範囲 `[start, end]` に重なる RecentClips セグメントは隔離しない。フラグ済みの瞬間（事故・接近など）の前後文脈を守る。
3. これら2ガード ＋「即削除せず隔離（N 日復旧窓）」で、走行中・低モーション事故の取りこぼしリスクを多層で抑える。

### 3.4 trash とマニフェスト
- **trash 先**: 共有ルート `@dcwb_trash/`（`@` 始まりで QNAP バックアップ対象から外しやすく、同一ファイルシステム内なので move は instant rename）。元の相対パス構造（`RecentClips/<date>/<file>`）を保持。
- **manifest**: `@dcwb_trash/manifest.jsonl`。1隔離 = 1行（append-only）。各行に `id` / `original_path`（共有ルート相対）/ `trash_path` / `segment_time`（JST）/ `quarantined_at`（UTC ISO）/ `motion_score` / `status`（`quarantined`｜`purged`｜`restored`）。
- **purge**: `quarantined_at` が `retention_days`（既定 14）を超えた `quarantined` 行を実削除し、`status` を `purged` に更新。
- **restore**: 指定 `id`（または `all`）の `quarantined` 行を元の相対パスへ move し戻し、`status` を `restored` に更新。元パスに同名ファイルが既存なら衝突として skip し警告。

### 3.5 ffmpeg_wrap 拡張
- 既存 `extract_frame(clip, t)` はフレーム1枚=ffmpeg 1起動。セグメント多数 × 8 フレームでは起動コストが嵩むため、**1クリップ1パスで複数フレームを抽出するヘルパ**を追加（例 `extract_frames(clip, times) -> list[np.ndarray]`、または `fps` サンプリング）。`calibrate` 等の既存利用はそのまま。

---

## 4. 設定 `pipeline.json`
`prune` セクションを追加（`render`/`verify` の B レイヤ設定と同居、`--pipeline-config` で差し替え可能）。

```json
{
  "prune": {
    "motion_threshold": 2.0,
    "frames_sampled": 8,
    "cameras_analyzed": ["front"],
    "min_age_hours": 48,
    "retention_days": 14,
    "trash_dir": "@dcwb_trash"
  }
}
```
`motion_threshold` の初期値は合成映像と実クリップで較正する（実装時にドライランで分布を見て確定）。

---

## 5. データフロー（隔離 1 サイクル）

```
1. scan: RecentClips を day-dir ごとに segment 化
2. guard: min-age / overlap で除外セグメントをフィルタ
3. score: 残りセグメントの front クリップから motion score 算出
4. select: score < threshold を隔離候補に
5. (dry-run) レポート出力で終了
   (--apply) 候補6カメラを @dcwb_trash へ move、manifest 追記、続けて purge 実行
```

---

## 6. テスト方針
既存の合成 mp4 フィクスチャ（`tests/fixtures/make_synthetic.py`、`libx264`）を流用し、以下を回帰テスト化する。

- **motion score**: 静止クリップ（全フレーム同一）は低スコア、動きのあるクリップは高スコア。閾値境界での候補選定。
- **segment 化**: 同一 `<ts>` の複数カメラが1セグメントにまとまる。
- **min-age ガード**: 直近セグメントが除外される（現在時刻を注入してテスト）。
- **overlap ガード**: SentryClips/SavedClips の時間範囲に重なるセグメントが除外される。
- **隔離 move**: tmp 上で候補が trash へ移り、manifest 行が正しく追記される。元ファイルが消えること。
- **purge by age**: `quarantined_at` が古い行だけ実削除＆ `status=purged`。新しい行は残る。
- **restore**: 隔離物が元パスへ戻り `status=restored`。衝突時 skip。
- **非対象保護**: SentryClips/SavedClips のファイルが一切変更されない。

---

## 7. 非対象（YAGNI）
- 自動スケジューリング（launchd / cron）— まず手動 CLI で振る舞いを観察してから、必要なら別途追加。
- serve UI への組み込み — 手動 CLI 優先。将来 `prune` モジュールを UI から呼べる粒度にはしておく。
- 位置情報（geofence / 画像照合）による自宅ガレージ特定 — モーション判定で要件を満たすため不要。
- `drive-data.json` との時刻×位置突合 — 現設計では使わない。
- QNAP `@Recycle` / スナップショットへの依存 — 復旧網として非活性のため採用しない。
