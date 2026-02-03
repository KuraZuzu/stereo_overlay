# 単眼2カメラ（上下配置）PoC：理想モデルでパネル位置をA映像にオーバレイ

本プロジェクトは、**上下に配置した2台のUVC単眼カメラ**を用いて、  
下側カメラ（B）で検出した **上下LEDバー付きパネル** の位置を、  
上側カメラ（A）の映像上に **リアルタイムで重ね描画** するためのサンプル実装です。

本READMEは **PoC（概ね合っていればOK）** を前提に、

- **同時撮影（ペア撮影）によるステレオキャリブレーションを行わない**
- **外部パラメータ（B→A）は理想モデル（設計値）で与える**

という運用に合わせた手順にしています。  
精度が必要になった場合は、後半の「精度を上げたい場合（オプション）」を参照してください。

---

## 1. システム概要

### カメラ構成

- **カメラA（上）**
  - 映像表示用
  - Bで推定したパネル位置を投影してオーバレイ表示

- **カメラB（下）**
  - パネル認識用
  - 上下LEDペアを検出し、縦画素長から距離を推定

### 前提（PoCの理想モデル）

- カメラA/Bは地面に対して概ね垂直に設置（大きなロール・ピッチ無し）
- カメラ間の**上下距離（ベースライン）**は既知：`baseline [m]`
- 外部パラメータは次を仮定（OpenCVカメラ座標：x=右, y=下, z=前）
  - 回転：`R = I`
  - 並進：`t = [0, ±baseline, 0]^T`
- Depthセンサーは不使用（パネル縦サイズ既知から距離推定）

---

## 2. ディレクトリ構成

```
stereo_overlay/
├── capture_calib_images.py      # キャリブレーション画像取得（単体/ペア）
├── calibrate_intrinsics.py      # 各カメラの内部パラメータ推定
├── calibrate_extrinsics.py      # 外部パラメータ生成（ideal / stereo）
├── run_overlay.py               # 実行プログラム
└── calib/
    ├── A/                        # カメラA画像
    ├── B/                        # カメラB画像
    ├── intrinsics_A.yaml
    ├── intrinsics_B.yaml
    └── extrinsics_B_to_A.yaml
```

> PoC運用では `calib/extrinsics_B_to_A.yaml` は **理想モデル**として生成します（ペア撮影不要）。

---

## 3. 環境構築

### 必要環境
- Ubuntu 20.04 / 22.04 / 24.04
- Python 3.8+
- UVC対応USBカメラ（`/dev/video*`）

### ライブラリ
```bash
pip install opencv-python numpy
```

---

## 4. PoC（理想モデル）での手順

### 4.1 単体撮影（A / B）

内部パラメータ（intrinsics）用に、各カメラでチェッカーボードを撮影します。  
この段階では **治具・ロボットへの取り付け不要**です。

実行時は、撮影する際のデバイスファイルと撮影画像出力先を指定します。

#### カメラAの撮影
```bash
python3 capture_calib_images.py --mode single --device /dev/video0 --out_dir calib/A
```

#### カメラBの撮影
```bash
python3 capture_calib_images.py --mode single --device /dev/video2 --out_dir calib/B
```

#### 解像度/FPS/FourCC を指定したい場合
```bash
python3 capture_calib_images.py --mode single --device /dev/video0 --out_dir calib/A --width 1280 --height 720 --fps 60 --fourcc MJPG
```

#### オプション

カメラAとカメラBで同時に撮影もできます。

```bash
python3 capture_calib_images.py --mode pair --deviceA /dev/video4 --deviceB /dev/video6 --outA calib/A --outB calib/B
```

解像度/FPS/FourCC を個別に指定する例：

```bash
python3 capture_calib_images.py --mode pair --deviceA /dev/video5 --deviceB /dev/video7 --outA calib/A --outB calib/B --widthA 1280 --heightA 720 --fpsA 60 --fourccA MJPG --widthB 1280 --heightB 720 --fpsB 60 --fourccB MJPG
```

**ポイント**
- 解像度・画角（ROI/クロップ）が実行時と同一になる設定で撮影する
  - 例：1280x720, 30fps （FPSによって画角が変わるものカメラもあるので注意）
- チェッカーボードは様々な位置・角度で撮る（目安：有効検出10枚以上）

---

### 4.2 内部パラメータ推定（A/B）

チェッカーボードの **内点数** と **マス目サイズ** を指定します。  
例：内点 9x6、1マス 0.025m の場合。

#### カメラA
```bash
python3 calibrate_intrinsics.py   --img_dir calib/A   --out calib/intrinsics_A.yaml   --board_cols 7   --board_rows 4   --square_size 0.09
```

#### カメラB
```bash
python3 calibrate_intrinsics.py   --img_dir calib/B   --out calib/intrinsics_B.yaml   --board_cols 7   --board_rows 4   --square_size 0.09
```

---

### 4.3 外部パラメータ生成（理想モデル：ペア撮影不要）

PoCでは、B→A変換を「上下距離だけ既知」「回転なし」と仮定して生成します。

例：AとBのレンズ中心の上下距離が 0.35m の場合：

```bash
python3 calibrate_extrinsics.py   --mode ideal   --baseline 0.35   --out calib/extrinsics_B_to_A.yaml
```

#### baseline_sign について（上下が逆に見える場合）
OpenCVのカメラ座標は **yが下方向**です。  
「AがBより上」なら、理想的には `t_y = +baseline` が基本ですが、取り付けや座標の取り方で逆に見える場合があります。

その場合は符号だけ反転します：

```bash
python3 calibrate_extrinsics.py   --mode ideal   --baseline 0.35   --baseline_sign -1   --out calib/extrinsics_B_to_A.yaml
```

> PoCではまず **baseline_sign の切替**だけで見た目を合わせるのが手軽です。

（必要なら `--roll/--pitch/--yaw` で手動微調整もできますが、PoCでは通常不要です）

---

### 4.3b 外部パラメータ生成（ペア撮影から推定：stereo）

同時撮影したペア画像から、B→A の外部パラメータを推定します。  
**ペア画像（`pair_XXXX_A.png` / `pair_XXXX_B.png`）が10組以上**あることが前提です。

```bash
python3 calibrate_extrinsics.py   --mode stereo   --dirA calib/A   --dirB calib/B   --intrA calib/intrinsics_A.yaml   --intrB calib/intrinsics_B.yaml   --out calib/extrinsics_B_to_A.yaml   --board_cols 7   --board_rows 4   --square_size 0.09
```

**ポイント**
- `--board_cols/--board_rows` は **チェッカーボードの内点数**（セル数は+1）
- A/Bの解像度は **同一を推奨**（違う場合は警告が出ます）

---

### 4.4 実行（リアルタイム）

#### パネル縦サイズの指定
`H` は「上下LED中心間距離 [m]」など、検出しやすい縦の実寸値を指定してください。

例：上下LED中心間距離が 0.180m の場合：

```bash
python3 run_overlay.py   --devA /dev/video5   --devB /dev/video7   --H 0.180
```

#### 実行中の表示
- **Camera B（デバッグ）**
  - 上下LED検出（緑）
  - 重心（赤）
- **Camera A（表示）**
  - 推定パネル位置（赤丸）
  - 推定距離 `Z` をテキスト表示

`q` キーで終了します。

---

## 5. 処理フロー（PoC）

1. カメラBで上下LEDを検出
2. 縦画素距離 `h_px` を算出
3. 単眼距離推定  
   `Z = (f_y * H) / h_px`
4. B画素 → 3D点へ逆投影（推定Zを使用）
5. **理想外部パラメータ**で B→A 変換（R=I, t=[0,±baseline,0]）
6. カメラA画像へ再投影して描画

---

## 6. 精度を上げたい場合（オプション）

PoCでは外部パラメータを理想モデルで仮定していますが、精度が必要になったら以下が有効です。

### 6.1 ペア撮影（A|B同時）でステレオキャリブレーション
治具・ロボット搭載後など「**最終の取り付け状態**」で、チェッカーボードをA/B同時に撮影します。

```bash
python3 capture_calib_images.py   --mode pair   --devA /dev/video0   --devB /dev/video2   --outA calib/A   --outB calib/B
```

### 6.2 stereoモードで外部パラメータ推定
```bash
python3 calibrate_extrinsics.py   --mode stereo   --dirA calib/A   --dirB calib/B   --intrA calib/intrinsics_A.yaml   --intrB calib/intrinsics_B.yaml   --out calib/extrinsics_B_to_A.yaml   --board_cols 9   --board_rows 6   --square_size 0.025
```

この `extrinsics_B_to_A.yaml` に差し替えるだけで、`run_overlay.py` はそのまま精度向上できます。

---

## 7. 実運用向け改善ポイント（任意）

- LED検出ロジックを既存の高精度版に差し替え（`detect_led_pair()`）
- 距離推定値 `Z` を移動平均 / IIR / カルマンフィルタで平滑化
- 歪み補正を `initUndistortRectifyMap` に置き換えて高速化
- 動体での遅延が気になる場合はタイムスタンプ同期を追加

---

## 8. 注意事項

- 解像度変更時は **内部パラメータが変わる**ため再キャリブレーション推奨
- 広角レンズでは歪み補正が重要（特に縦画素長→距離推定に影響）
- 遠距離では `h_px` が小さくなり、距離推定誤差が増えます（平滑化推奨）

---

## 9. 想定用途

- 移動ロボット搭載パネルの大まかな可視化（PoC）
- センサ非搭載環境での簡易位置推定
- ロボットUI向けのカメラオーバレイ表示
