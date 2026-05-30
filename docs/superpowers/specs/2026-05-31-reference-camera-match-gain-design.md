# 参照カメラ・マッチゲインによる B レイヤ差し替え

設計日: 2026-05-31 / 対象: `dcwb` の色補正（B レイヤ）拡張
関連: `docs/adr/0001-consumer-camera-reference-color-matching.md`（目的の決定）、
`docs/superpowers/specs/2026-05-09-tesla-dashcam-wb-design.md`（A×B 補正モデル）、
`docs/superpowers/specs/2026-05-30-insta360-tesla-sync-design.md`（映像差分による時刻同期）

## 目的

一般カメラ（Insta360 / iPhone 等）の映像と Tesla 6 カメラ映像を合成しても**違和感なく見える**よう、
Tesla の色補正 **B レイヤ（シーン光）を「参照カメラとのマッチゲイン」に差し替える**機能を追加する。
参照カメラが無い場合は**従来どおり**（gray-world の Shades-of-Gray 推定）。

目的は測色的正確さ（true D65）ではなく**合成時の知覚的整合**であり、参照カメラが較正済み基準でない
ことは許容する（ADR 0001）。

## 全体データフロー

```
[算出] dcwb match-reference <参照動画> --recent <date> [--write]
         → 映像差分で参照を Tesla front に時刻同期
         → 同期した昼間・走行フレーム対で「A適用後Tesla」と「参照」の Shades-of-Gray を取り比を幾何平均
         → 参照ゲイン (g_r, 1, g_b) を標準出力、--write 指定で pipeline.json の awb.reference_gain に書込み
                              │
                              ▼
[補正] dcwb render / render-all
         → awb.reference_gain があれば B = その固定ゲイン（6 カメラ・全クリップ一律）
         → 無ければ従来の estimate_scene_gain（per-clip gray-world、挙動不変）
```

参照ゲインは scene/光源依存なので、合成対象ドライブごとの `pipeline.json` に置く想定。

## コンポーネント

### 1. `src/dcwb/refmatch.py`（新規）

参照ゲインの算出を担う。1 つの責務＝「参照動画 + Tesla front → 参照マッチゲイン (g_r,1,g_b)」。

- `reference_gain(tesla_sog, ref_sog) -> tuple[float, float, float]`（純関数）
  - 入力は両カメラの Shades-of-Gray ゲイン（緑正規化済み `(g_r,1,g_b)`、`awb.shades_of_gray` の戻り値形式）。
  - `awb.shades_of_gray` は「そのカメラを中性化するゲイン」を返す。A 適用後 Tesla の中性化ゲインを
    `g_T`、参照の中性化ゲインを `g_R` とすると、**A 適用後 Tesla を参照の発色に寄せるゲイン
    `G = g_T / g_R`（チャンネル毎、緑正規化して `g_g≡1`）** を返す。
    - 導出: A 適用後 Tesla の残差イルミナントは `1/g_T`、参照は `1/g_R`。Tesla にゲイン `G` を乗じると
      残差イルミナントが `G/g_T` になり、これを参照の `1/g_R` に一致させるには `G = g_T/g_R`。
  - 参照が完全中性（`g_R=(1,1,1)`）なら `G=g_T`＝従来の gray-world 中性化に一致（連続性）。

- `compute_reference_gain(reference, fronts, front_profile, *, source, samples, ...) -> tuple[float,float,float]`
  （オーケストレーション）
  - 手順:
    1. `sync._detect_visual_offset(reference, tesla_concat, ...)` で参照を Tesla front に時刻同期
       （純画素フレーム差分の相互相関。`.insv` は前レンズ生魚眼 `[0:v:0]`、平面 mp4 は `0:v:0`）。
       Tesla 側は `sync.select_front_clips` で対象 front クリップを選び `ffmpeg_wrap.concat_clips` で連結。
    2. 同期した**昼間・走行中**の複数時刻でフレーム対を抽出（既定 ~8〜10 点、`daylight.is_daytime`）。
       - 参照が `.insv`: `ffmpeg_wrap.reframe_insv` で前方平面にリフレーム（sync の既定画角）してから抽出。
       - 参照が平面 mp4: そのまま抽出。
       - 双方とも**中央の外景クロップ**（Insta の車内/Tesla のボンネットを避ける固定割合クロップ）。
    3. Tesla front フレームに **front の A（`profile.matrix_3x3`）を適用**（`flat @ A.T`、0–1 クリップ）
       してから `awb.shades_of_gray` → `g_T`。参照フレームは素のまま `awb.shades_of_gray` → `g_R`。
    4. フレーム毎に `reference_gain(g_T, g_R)` を求め、**幾何平均**（乗算的ゲインに頑健）で集約。
  - 失敗時（同期できない/昼間フレーム無し）は明確な例外を投げ、CLI が分かりやすく報告。

### 2. `dcwb match-reference` サブコマンド（`cli.py`）

- 引数: `reference`(Path)、`--recent <YYYY-MM-DD>`(必須)、`--source`(既定 `/mnt/sentryusb`)、
  `--profiles-dir`(既定 `profiles`)、`--pipeline-config`(既定 `pipeline.json`)、
  `--write`(指定で pipeline.json の `awb.reference_gain` を更新)、`--samples`(既定 10)。
- 動作: front profile を読み、`refmatch.compute_reference_gain` を呼び、結果を
  `[g_r, 1.0, g_b]` 形式で標準出力（人間可読＋JSON 1 行）。`--write` 時は pipeline.json を読み込み
  `awb.reference_gain` を設定して書き戻す（他キーは保持）。

### 3. render の B レイヤ消費（`render.py`）

- `render_event`: `awb_cfg.get("reference_gain")` が真なら、`estimate_scene_gain` を呼ばず
  **全クリップで `scene_gain = tuple(reference_gain)`** とする。無ければ従来どおり per-clip 推定。
- 合成は既存 `compose_clip_matrix(profile, scene_gain, gain_min, gain_max, attenuation)` を**そのまま再利用**:
  `final = diag(scene_gain) @ profile.matrix_3x3`。A は各カメラのままなので 6 カメラが参照トーンに揃う。
- **安全装置は既存挙動を流用**（特別扱いを増やさない・1 経路に統一）:
  - `gain_min/max`（既定 0.7–1.5）外なら B 破棄→A のみ（妥当な参照ゲインなら発火しない安全網）。
  - 夜間 `attenuation`（既定 0.5）も従来どおり参照ゲインに適用（夜は参照トーンも恒等へ弱める）。
- `_pipeline.json` スナップショットに採用 `reference_gain` を記録（`scene_gain` 欄が参照値になる）。

### 4. 設定 `pipeline.json`

`awb` セクションに `reference_gain: [g_r, g_g, g_b]`（未設定 or `null` で従来動作）を追加。
README に「scene/光源依存の値であり、合成対象ドライブごとの config に置く」「`match-reference --write` で生成」を明記。

## エラー処理・エッジケース

- 参照ゲインが極端（`gain_min/max` 外）: render 既存の安全網で A のみにフォールバック（B 破棄）。
  `match-reference` 側でも妥当域外なら警告を出す。
- 参照の同期が弱い（フレーム差分 peak が低い）: `match-reference` が警告し、それでも値は出す
  （ユーザ判断。`sync` と同じ思想）。
- 昼間フレームが無い（夜間ドライブ）: 算出を中断しエラー（B 差し替えの前提が崩れるため）。
- 参照ゲイン未設定時は **既存テスト・挙動が一切変わらない**こと（回帰防止）。

## テスト

- `reference_gain()` 純関数: 既知の `g_T`,`g_R` から比＋緑正規化が正しいこと。参照中性なら `g_T` に一致。
- `render_event`/`compose_clip_matrix` の参照経路: `awb.reference_gain` 設定時に最終行列が
  `diag(ref) @ A` になること、未設定時は従来どおり（合成 mp4 で回帰）。クランプ・夜間減衰の相互作用。
- `match-reference` cli 配線: 重い同期/抽出は monkeypatch し引数転送と `--write` の pipeline.json 更新を検証。
- 実データ smoke（手動）: `match-reference <insv> --recent 2026-05-27` がゲインを出し、`--write` 後の
  `render` が参照トーンで補正することを目視確認。

## モジュール依存

`refmatch.py` → `sync`（`_detect_visual_offset`/`select_front_clips`）, `awb.shades_of_gray`,
`ffmpeg_wrap`（`concat_clips`/`reframe_insv`/フレーム抽出）, `profile`, `daylight`。
`cli.py` に `match-reference` 追加。`render.py` の `render_event` を参照ゲイン対応に小改修。

## 非対象（YAGNI）

- カメラ別の参照ゲイン（6 カメラは一律）。
- 参照側トーン（S 字・彩度＝look）まで合わせる処理（本機能は WB ゲインのみ）。
- 夜間・混在光源向けの特別な算出。
- 複数ドライブをまたいだ参照ゲインの一般化。
