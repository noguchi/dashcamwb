# Insta360 乗車視点 × Tesla ドラレコ フレーム精度同期（`dcwb sync-insta360`）

設計日: 2026-05-30 / 対象: `dcwb` CLI 拡張

## 目的

Insta360 で録画した乗車視点映像と、Tesla Model 3 Highland のドラレコ front 映像を
**フレーム精度で時間同期**し、次の 2 つを出力する再利用可能なサブコマンド `dcwb sync-insta360` を追加する。

- **(B) 合成 mp4** — Insta360 乗車視点と Tesla front を**横並び**、Tesla SEI（速度・ステア角・ギア）を
  `drawtext` で焼き込んだ単一動画（ブレインストームのレイアウト案 D）。
- **(C) ローカル Web プレイヤー** — 両動画を同時再生し、片方をスクラブすると他方が追従。
  手動ナッジで最終オフセットを微調整できる。

「完全に同期」を目標とし、自動同期を狙いつつ人手で確実に補正できる保険を持つ。

## 入力と前提

```
dcwb sync-insta360 <insv ...> --recent <date> [--insta-flat <mp4>] [--out-root <dir>]
```

- `<insv ...>` — Insta360 原版（例 `VID_..._007.insv VID_..._008.insv VID_..._009.insv`、グロブ可）。
  - デュアル魚眼 HEVC（2×3008×3008）＋ AAC、1 セグメント ≈ 30 分、29.97fps。
  - `creation_time`（mp4 ヘッダ、**UTC**。例 `2026-05-27T08:17:57Z` = 17:17:57 JST）が**カメラ RTC の絶対時刻アンカー**。
  - GPS/IMU は標準 data トラックではなく**末尾の Insta360 独自トレーラ**（例 offset 0x449407f26、約 99MB）に格納。
- `--recent <date>` — `RecentClips/<date>` の front クリップ群（1 分セグメント）。Insta360 開始時刻を含む
  時間窓を自動抽出する。Tesla 側は SEI にフレーム単位の `vehicle_speed_mps` / `latitude/longitude` /
  `heading_deg` / `steering_wheel_angle` / `linear_acceleration_*` / `gear_state` を持ち、**音声は無い**
  （→ 音声相関は使えない）。ファイル名の壁時計は GPS 同期で正確（naive JST と解釈、`calibrate.JST`）。
- `--insta-flat <mp4>` — 表示用にユーザが Insta360 Studio で書き出した平面「乗車視点」mp4（任意）。
  未指定なら ffmpeg `v360`（dfisheye→平面）で自動リフレームして表示映像を生成する。

`.insv` 自体は**テレメトリと絶対時刻の供給源**であり、表示には使わない。

## 同期方式: ハイブリッド（アンカー → 相互相関 → 手動ナッジ）

### 段 1 — 時刻アンカー（粗）
insv `creation_time`(UTC→JST) と Tesla ファイル名(JST) から粗オフセット `t0` を計算し、対象 Tesla 区間を確定。
カメラ RTC は GPS 同期されておらず数秒ズレうるため、これは探索の初期値に留める。

### 段 2 — テレメトリ抽出
- **Tesla** — 既存 `telemetry`（vendored `tesla_dashcam` SEI 抽出）を時系列化し、
  `speed` / `heading→ヨーレート d(heading)/dt` / `accel_x` を得る。
- **Insta360** — トレーラ（末尾約 99MB）に `seek` して **IMU(gyro/accel) を時系列化**する。
  18GB 本体は読まない。Insta360 トレーラの IMU フォーマットは gyroflow の `telemetry-parser` が
  リバースエンジニアリング済みで、これを移植する（`vendor/` に NOTICE 付きで取り込む。Tesla SEI を
  vendored した前例に倣う）。

### 段 3 — 精密同期（フレーム精度）
共通の 1 次元モーション信号を `t0 ± 窓`(既定 ±10s) で**相互相関**し、相関を最大化するオフセット `δ` を求める。
- **既定信号 = ヨーレート**（Tesla `d(heading)/dt` vs Insta360 `gyroZ`）。曲がり動作が鋭く相関ピークが立つ。
- **フォールバック = 前後加速度**（Tesla `linear_acceleration_x` vs Insta360 加速度 X）。
- 両座標系の**軸・符号・単位を較正**してから相関を取る。相関ピーク値を**信頼度**として記録する。

### 段 4 — リフレーム timeline マッピング
同期計算は insv-timeline 上で行う。`--insta-flat`（リフレーム書き出し）を表示に使う場合、
「書き出し t=0 ＝ insv 先頭セグメント(007)の creation_time」（**無トリミング**）を前提とし、
**書き出し尺 ≈ insv 全セグメント尺の合計**で自動検証する。ズレを検出したら警告し、手動ナッジで吸収する。
v360 自動リフレームの場合は identity（insv-timeline と一致）。

## 出力

`<out-root>/sync/<date>/` に以下を生成する（`<out-root>` 既定はレンダ出力と同じ規約）。

- `combined-<...>.mp4` — **(B)**。ffmpeg で 表示Insta360（リフレーム or v360）と Tesla front concat を横並び、
  `δ` で時間整列、Tesla SEI（速度/ステア角/ギア）を `drawtext` でオーバーレイ（レイアウト D）。
  Tesla 側は既存 A×B 白色補正行列＋look グレードを流用可能。
- `sync.json` — `δ`・信頼度・座標較正・telemetry 時系列・各動画パス・尺検証結果を記録する manifest。
- 抽出 IMU/SEI CSV — デバッグ・回帰テスト用。
- **(C)** Web プレイヤーは `serve/` に同期再生ページとして追加し、`sync.json` を読み込む。
  master `<video>` に slave を `currentTime + δ` で追従させ、**手動ナッジスライダ**で `δ` を補正して
  `sync.json` に書き戻す。telemetry は JSON からライブ描画（焼き込み不要）。

## 性能方針

NVMe(`./`,`/home`: 書き 2.6 / 読み 6.3 GB/s, 空き 410G, 永続) ≫ CIFS(`/mnt/sentryusb`: 書き 253 / 読み 148 MB/s)
≫ ※ `/tmp` は tmpfs(RAM, 30G 上限・揮発) で .insv 37G を載せきれない。

- IMU 抽出は**トレーラ seek のみ**（CIFS 上で完結、コピー不要）。
- v360 自動リフレームで insv 全読みが必要な場合のみ、**NVMe 作業ディレクトリにステージング**する
  （`/tmp` tmpfs は使わない）。
- 合成 mp4 レンダリングは表示用 mp4（軽量）と Tesla 小クリップだけを読み、18GB insv は読まない。

## モジュール構成

- `insta360.py`（新規） — トレーラ IMU パーサ ＋ `creation_time` 読取り。
- `sync.py`（新規） — 時刻アンカー、信号較正、相互相関、信頼度算出。
- `ffmpeg_wrap.py`（拡張） — 横並び合成 ＋ telemetry `drawtext` オーバーレイ。
- `serve/`（拡張） — 同期プレイヤー route ＋ template。
- `cli.py`（拡張） — `sync-insta360` サブコマンド。
- `vendor/`（追加） — Insta360 トレーラ IMU 仕様（gyroflow 由来、NOTICE 明記）。

## リスクとフォールバック

- **最大リスク = Insta360 IMU パース**（独自フォーマット）。**最初のマイルストーンを抽出 spike** とし、
  使える gyro/accel 時系列が取れることを先に証明する。失敗時は段 3 を省き **「アンカー＋手動ナッジ」に縮退**
  （自動相関なしでも完全同期は手動で到達可能）。
- **GPS** は原版に無い可能性（Insta360 X 系は paired デバイスが必要）。あれば Tesla `lat/lon` 軌跡との
  照合を**ボーナス経路**として追加し、対象区間の自動特定にも使える。
- **リフレーム書き出しのトリミング**で timeline がズレるケースは尺検証で検出し、手動ナッジで救済。

## テスト

- 既知 `δ` を注入した合成モーション信号での相関ピーク検出（`sync.py` 単体）。
- 時刻アンカー計算、リフレーム尺検証、座標系較正の単体。
- serve プレイヤーの `δ` 往復（書き戻し）。
- 合成 mp4 レンダの回帰（合成 libx264 mp4、CI の `h264_videotoolbox→libx264` フォールバックに依存）。

## 参考

- 既存設計: `docs/superpowers/specs/2026-05-29-vlm-drive-highlight-design.md`（highlight パイプライン・look グレード）、
  `docs/superpowers/specs/2026-05-29-telemetry-gear-prune-design.md`（Tesla SEI 利用）。
