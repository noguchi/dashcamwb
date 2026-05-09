# Tesla DashCam ホワイトバランス補正パイプライン 設計仕様

- **作成日**: 2026-05-09
- **対象**: 単一の Tesla Model 3 Highland 車両（所有1台）から得られる DashCam 映像のホワイトバランス補正
- **位置づけ**: 設計仕様（design spec）。実装計画は別途 `writing-plans` で作成する。

---

## 1. 目的とスコープ

### 1.1 達成したいこと
Tesla Model 3 Highland の DashCam 映像（front / back / left_pillar / right_pillar / left_repeater / right_repeater の6カメラ）を、**D65 (≒6500K) で撮影した sRGB ニュートラル**な見た目に補正するパイプラインを構築する。

- 白い対象がグレースケール上で R=G=B（カラーキャストなし）になる
- 6カメラ間で同一被写体の色が揃う
- シネマ風グレーディング等の表現は本パイプラインの責務外（中性化までで完了）

### 1.2 ユースケース
過去および今後の全 DashCam 素材（QNAP に蓄積、約360GB / 162 Sentry イベント / 12,558 mp4）に流すパイプライン。1イベント単位でユーザが CLI を叩いてレンダー、または一括処理。

### 1.3 入力の規模と分布
- **保存先**: `/Volumes/sentryusb/{SentryClips,RecentClips,SavedClips}/`
- **イベント数**: SentryClips 162 件
- **時間分布**（Sentry イベント発生時刻）:
  - 純粋な昼間 (7–16時): 約 70%
  - 黄昏 (17–19時): 約 20%
  - 夜間 (20–6時): 約 10%

### 1.4 補正手法の方針
- **レイヤー A**: カメラ素性補正（per-camera fixed correction、3×3対角行列または将来的にCCM）
- **レイヤー B**: シーン適応 AWB（per-clip illuminant estimation, Shades of Gray p=6）
- A をベースに、B を上に重ねたハイブリッド構成。
- カラーチャート（ColorChecker）は採用しない。Tesla DashCam の H.264 8bit 出力に対して精度過剰、かつ6カメラの正面提示の物理的手間が大きい。

---

## 2. アーキテクチャ全体像

```
┌─────────────────────────────────────────────────────────────────┐
│ /Volumes/sentryusb (QNAP, 読み取り専用)                          │
│ - SentryClips/<timestamp>/<6 cameras × N segments>.mp4          │
│ - RecentClips/<date>/...                                        │
│ - SavedClips/...                                                │
└──────────────┬──────────────────────────────────────────────────┘
               │
   ┌───────────▼─────────────┐         ┌─────────────────────┐
   │  calibrate.py           │ writes  │ profiles/           │
   │  (statistical mining)   ├────────▶│  front.json         │
   │   - sample N frames     │         │  back.json          │
   │   - detect neutral      │         │  left_pillar.json   │
   │     patches             │         │  right_pillar.json  │
   │   - robust median       │         │  left_repeater.json │
   │     RGB ratio           │         │  right_repeater.json│
   └─────────────────────────┘         └──────────┬──────────┘
                                                  │
                              ┌───────────────────▼───────────┐
   user CLI                   │  render.py                    │
   $ dcwb render <event> ────▶│   per camera:                 │
                              │    1. estimate scene AWB (B)  │
                              │       on N sample frames      │
                              │    2. compose final 3×3       │
                              │       = scene_gain × profile  │
                              │    3. ffmpeg one-pass with    │
                              │       colorchannelmixer +     │
                              │       VideoToolbox encode     │
                              └────────────┬──────────────────┘
                                           │
                                           ▼
                /Users/noguchi/AI/dashcamwb/corrected/<event>/
                  - 6 corrected mp4s
                  - event.json (copy)
                  - thumb.png (corrected)
                  - _pipeline.json (used profile + AWB values)
```

### 2.1 主要設計判断

- **元データは読み取り専用**。インプレース置換は禁止。
- **遅延レンダー**戦略: 補正値は `profiles/*.json` に永続化、出力は CLI 起動時に都度生成。全件並列レンダーはせず、ストレージを2倍にしない。
- **クリップ単位定数 B**: 1分前後のクリップ内では照明変化を無視し、クリップごとに B を1回計算 → A と合成して**単一の3×3行列**として ffmpeg に渡す。時変フィルタは初期スコープ外。
- **出力先**: `/Users/noguchi/AI/dashcamwb/corrected/<event_name>/`（決め打ち）。
- **実装スタック**: Python + OpenCV + ffmpeg。Apple Silicon の VideoToolbox H.264 ハードウェアエンコードを利用。

---

## 3. キャリブレーション（statistical mining）

`calibrate.py` の責務: 全 162 イベントから各カメラの D65 白点を推定し、`profiles/<camera>.json` を生成する。

### 3.1 昼間判定
イベント発生時刻 (`event.json["timestamp"]`) と緯度経度（`event.json["est_lat"]`, `est_lon"]` 優先、なければ Tokyo: 35.6762°N, 139.6503°E）から `astral` ライブラリで日の出/日の入りを計算。

- サンプル対象: **日の出 + 30分 〜 日の入り - 30分** に発生したイベントのクリップのみ
- 範囲外（夜・薄明）は calibrate からは除外（B レイヤーで対応）

### 3.2 サンプリング手順（カメラごとに独立に実行）
1. 昼間判定で対象となった全クリップから `ffmpeg -ss <t> -frames:v 1` で1イベントあたり最大 3 フレーム抽出
2. **ニュートラル候補マスク生成** (OpenCV、HSV 空間):
   - 高輝度かつ低彩度: `V > 0.7 && S < 0.15`
   - 過飽和除外: `R or G or B > 250` を除外
   - 影除外: `V < 0.2` を除外
3. **空マスク追加検出**（front / repeater のみ）: 画像上半 1/3 内、低彩度、青寄りピクセルを別カテゴリで採取
4. **多色性チェック**: 1サンプル内の彩度分布の標準偏差が低いフレームは除外（赤い夕焼け・緑の森のような単色シーンを除く）
5. **集約**: 各カメラについて全サンプルの白パッチ (R/G, B/G) 比を集計、**幾何中央値**で外れ値耐性を確保
6. **ゲイン算出**: 白点 (Rw, Gw, Bw) から `gain_R = Gw/Rw`, `gain_G = 1`, `gain_B = Gw/Bw`
7. **検証出力**: `calibration_report.html` に推定値と適用例を出力、人間が目視確認

### 3.3 警告条件
- カメラごとに 50 サンプル未満しか集まらないイベント分布の場合 → 警告（`max-samples-per-event` を増やすか、calibrate を再実行）

### 3.4 プロファイル形式（`profiles/front.json` 例）
```json
{
  "camera": "front",
  "gain_r": 0.918,
  "gain_g": 1.000,
  "gain_b": 1.067,
  "matrix_3x3": [[0.918, 0, 0], [0, 1.0, 0], [0, 0, 1.067]],
  "calibration": {
    "samples_used": 247,
    "events_sampled": 89,
    "method": "robust_white_patch_median",
    "calibrated_at": "2026-05-09T12:34:56+09:00",
    "samples_per_event_max": 3
  }
}
```

3×3 行列形式で永続化することで、将来的に CCM（非対角成分を持つカラー補正行列）への置き換えが上位コードを変えずに可能。

---

## 4. シーン適応 AWB（B レイヤー）

`render.py` 内、各クリップのレンダー直前に1回実行。

### 4.1 アルゴリズム
**Shades of Gray (Minkowski p=6)** を採用。

- Gray World (p=1) より単色シーン耐性が高い
- White Patch (p=∞) より飽和ピクセル耐性が高い
- 推定式: `e_c = (∫ |f_c|^p)^(1/p) for c in {R,G,B}`、ゲイン: `g_c = max(e) / e_c`

### 4.2 手順（クリップごと）
1. クリップ全長から等間隔に **10 フレーム**を `ffmpeg -ss <t> -frames:v 1` で抽出
2. 各サンプルにカメラ profile の 3×3 を**先に**適用（カメラ素性除去後に純粋な照明色を推定するため）
3. 飽和（>0.97）と影（<0.03）の画素を計算から除外
4. Shades of Gray (p=6) を適用 → クリップ平均の照明色を得る
5. ゲイン算出 → **最終 3×3**: `final_matrix = diag(g_R, g_G, g_B) × profile.matrix_3x3`

### 4.3 安全装置（フォールバック）
- 推定ゲインが大きく振れる場合（任意のチャネルで `g < 0.7` または `g > 1.5`）→ B をゼロ強度に減衰、A のみ適用、ログ警告
- イベント発生時刻が日没後 → B の効きを 0.5 倍に弱める（人工光下では Shades of Gray が外しやすい）

### 4.4 設定ファイル `pipeline.json`
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
  }
}
```

---

## 5. レンダリングパイプライン

ffmpeg 1パス、Apple Silicon ハードウェアエンコード前提。

### 5.1 クリップごとのコマンド（疑似）
```bash
ffmpeg -hwaccel videotoolbox -i <input.mp4> \
  -vf "colorchannelmixer=\
    rr=<m11>:rg=<m12>:rb=<m13>:\
    gr=<m21>:gg=<m22>:gb=<m23>:\
    br=<m31>:bg=<m32>:bb=<m33>" \
  -c:v h264_videotoolbox -b:v 12M -profile:v high \
  -c:a copy \
  -movflags +faststart \
  <output.mp4.tmp> && mv <output.mp4.tmp> <output.mp4>
```

### 5.2 設計判断
- `h264_videotoolbox` で M1/M2/M3 専用エンコーダを使用（実時間以上の速度、CPU 数% で済む）
- ビットレート 12Mbps: Tesla オリジナル (約 6-10Mbps) からわずかに上振れさせて再エンコードのジェネレーション損失を最小化
- 音声は `-c:a copy`（無ければ無視される）
- `.mp4.tmp → mv` でアトミック化（途中失敗しても破損ファイルが残らない）
- `+faststart` で moov atom 先頭化、QuickTime/iOS で即時再生可能

### 5.3 並列処理
- VideoToolbox は同時 1〜2 セッションが現実的。`render.py` は逐次でOK
- 複数イベントを並列に流したい場合のみ外側で `xargs -P 2` 等

### 5.4 性能見積もり（Apple Silicon M2/M3）
- 1分クリップ × 1080p: 約 5〜10 秒
- 1イベント (6カメラ × 平均2分 = 12クリップ): 約 1〜2 分
- 162イベント全件レンダー: 約 3〜5 時間

---

## 6. CLI / UX

### 6.1 コマンド構成
`dcwb` コマンドを `pyproject.toml` の `[project.scripts]` で公開。

```bash
# キャリブレーション（profiles/*.json を生成 or 更新）
dcwb calibrate \
  --source /Volumes/sentryusb \
  [--max-samples-per-event 3] \
  [--report-out /Users/noguchi/AI/dashcamwb/calibration_report.html]

# レンダリング（イベント単位）
dcwb render <event_dir>
# 例: dcwb render /Volumes/sentryusb/SentryClips/2026-05-05_13-50-46
# 出力: /Users/noguchi/AI/dashcamwb/corrected/2026-05-05_13-50-46/

# 検証（補正前後を並べた HTML を出力）
dcwb verify <event_dir>

# 一括レンダー
dcwb render-all --source /Volumes/sentryusb/SentryClips
```

### 6.2 出力規約
`/Users/noguchi/AI/dashcamwb/corrected/<event_name>/` 直下に:
- 6 カメラの mp4（元と同名）
- `event.json`（コピー）
- `thumb.png`（補正後）
- `_pipeline.json`（このレンダーで使った profile + AWB 値のスナップショット、後追いデバッグ用）

### 6.3 ログ
- `stderr`: 進行表示
- `logs/<event>.log`: 詳細ログ（採用ゲイン値・フォールバック発動有無）

---

## 7. テストと検証戦略

### 7.1 単体テスト（pytest）
- `profile.py`: JSON ロード/保存の往復不変
- `awb.py`: 既知の合成画像（純グレー、空+地面、過飽和込み）に対するゲイン推定の数値テスト
- `matrix.py`: マトリクス合成（A × B）の結合則・恒等

### 7.2 結合テスト
- 合成テストイベント（合成 mp4 with known white object）でレンダーパイプライン全体を実行、出力の白パッチが R=G=B±誤差 になることを assert
- フィクスチャは `tests/fixtures/synthetic_event/` に配置、ffmpeg で生成するスクリプトを同梱

### 7.3 視覚検証（人間ループ）
- `dcwb verify <event>`: 補正前 / A だけ / A+B 適用後 を3列で並べた HTML を出力
- カメラ間で「同じ白い物体が同じグレーに見える」ことを目視確認
- キャリブレーション直後は 5〜10 イベントで verify を回して値を追い込む

### 7.4 エラーハンドリング
- 入力 mp4 が読めない → スキップ + 警告、他のクリップは続行
- profile が無いカメラ → エラー（calibrate 先行を要求）
- ffmpeg 失敗 → `.tmp` を削除して非ゼロ終了

### 7.5 監視値（ログから可視化用）
- クリップごとの推定ゲイン (g_R, g_B)
- B フォールバック発動率（高ければ B のパラメータが厳しすぎる）

---

## 8. リポジトリ構成（想定）

```
/Users/noguchi/AI/dashcamwb/
├── docs/
│   └── superpowers/specs/2026-05-09-tesla-dashcam-wb-design.md  (本文書)
├── src/
│   └── dcwb/
│       ├── __init__.py
│       ├── cli.py            # entry points
│       ├── calibrate.py      # statistical mining
│       ├── render.py         # rendering pipeline
│       ├── awb.py            # B layer (Shades of Gray)
│       ├── profile.py        # profile load/save
│       ├── matrix.py         # 3x3 composition
│       ├── ffmpeg_wrap.py    # ffmpeg 呼び出しラッパ
│       └── verify.py         # HTML report 生成
├── profiles/                 # キャリブレーション結果
│   ├── front.json
│   ├── back.json
│   ├── left_pillar.json
│   ├── right_pillar.json
│   ├── left_repeater.json
│   └── right_repeater.json
├── pipeline.json             # B レイヤー設定
├── corrected/                # 出力先
│   └── <event_name>/...
├── logs/
├── tests/
│   ├── fixtures/synthetic_event/
│   ├── test_profile.py
│   ├── test_awb.py
│   ├── test_matrix.py
│   └── test_render_integration.py
├── pyproject.toml
└── README.md
```

---

## 9. スコープ外（明示的に除外）

- カラーチャート（ColorChecker）を用いた精密 CCM
- フレーム時変 AWB（クリップ内で照明が動的変化する場合の補正）
- シネマ風グレーディング、彩度・コントラスト調整
- 自動シーン分類（昼/夜/トンネルの自動切替プロファイル）
- DashCam 映像の H.264 → ProRes 等への形式変換
- GUI / Web UI

これらは将来の拡張候補。本仕様の実装後、視覚検証で不満があれば追加検討する。

---

## 10. 実装で確認する事項

設計レベルで決定済みだが、実装中に実測値で詰める項目:

- **VideoToolbox の並列セッション数**: 現実的な上限を 1〜2 と見積もったが、M2/M3 上で 3〜4 まで上げて品質劣化やスループット飽和を実測してから決める
- **B レイヤーのフォールバック閾値 (`gain_min`/`gain_max`)**: 0.7〜1.5 を初期値とするが、162イベントで verify を回して fallback 発動率が異常に高い/低い場合は調整
- **profile キャリブレーション再実行のトリガ**: Tesla のファームウェア更新でカメラ色設計が変わった場合に再実行が必要。検出方法は本仕様外（運用上ユーザが手動で再 calibrate を判断）

### 確定済みの実装方針
- 依存ライブラリ: `numpy`, `opencv-python`, `astral`, `jinja2`（HTML report 用）。`pyproject.toml` の `dependencies` にすべて記載。
- VideoToolbox 品質設定: 12Mbps 固定ビットレート（`-b:v 12M`）。VBR/CRF は使用しない。
- ログテンプレート: Python 標準 `logging` モジュール、`jinja2` で HTML report 生成。
