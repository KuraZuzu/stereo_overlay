# 単眼2カメラ（上下配置）によるパネル位置推定・オーバレイ表示

本リポジトリは、**上下に配置した2台のUVC単眼カメラ**を用いて、  
下側カメラ（B）で検出した **上下LEDバー付きパネル** の位置を、  
上側カメラ（A）の映像上に **リアルタイムで重ね描画** するためのサンプル実装です。

Depthセンサーは使用せず、**パネルの縦サイズが既知であること**を利用して  
単眼画像から奥行きを推定します。

---

## 1. システム概要

### カメラ構成

- **カメラA（上）**
  - 映像表示用
  - Bで推定したパネル位置を投影してオーバレイ表示

- **カメラB（下）**
  - パネル認識用
  - 上下LEDペアを検出し、縦画素長から距離を推定

### 前提条件

- カメラA/Bは地面に対して垂直に設置
- カメラ間の距離は固定
- パネルはロボットに固定され、常に正立（ロール・ピッチしない）
- パネルの **上下LED中心間距離 H [m]** が既知

---

## 2. ディレクトリ構成

```
stereo_overlay/
├── capture_calib_images.py      # キャリブレーション画像取得
├── calibrate_intrinsics.py      # 各カメラの内部パラメータ推定
├── calibrate_extrinsics.py      # B→A 外部パラメータ推定
├── run_overlay.py               # 実行プログラム
└── calib/
    ├── A/                        # カメラA画像
    ├── B/                        # カメラB画像
    ├── intrinsics_A.yaml
    ├── intrinsics_B.yaml
    └── extrinsics_B_to_A.yaml
```

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

## 4. キャリブレーション手順

### 4.1 チェッカーボード画像の取得

#### 単体撮影（A / B）

```bash
python3 capture_calib_images.py   --mode single   --devA /dev/video0   --outA calib/A
```

```bash
python3 capture_calib_images.py   --mode single   --devA /dev/video2   --outA calib/B
```

#### ペア撮影（外部パラメータ用）

```bash
python3 capture_calib_images.py   --mode pair   --devA /dev/video0   --devB /dev/video2   --outA calib/A   --outB calib/B
```

> **ポイント**
> - 解像度・FPSは実行時と同一にする
> - チェッカーボードは様々な位置・角度で撮影する（最低10枚以上）

---

### 4.2 内部パラメータ推定（A/B）

```bash
python3 calibrate_intrinsics.py   --img_dir calib/A   --out calib/intrinsics_A.yaml   --board_cols 9   --board_rows 6   --square_size 0.025
```

```bash
python3 calibrate_intrinsics.py   --img_dir calib/B   --out calib/intrinsics_B.yaml   --board_cols 9   --board_rows 6   --square_size 0.025
```

- `square_size`：チェッカーボード1マスの実寸 [m]

---

### 4.3 外部パラメータ（B→A）推定

```bash
python3 calibrate_extrinsics.py   --dirA calib/A   --dirB calib/B   --intrA calib/intrinsics_A.yaml   --intrB calib/intrinsics_B.yaml   --out calib/extrinsics_B_to_A.yaml   --board_cols 9   --board_rows 6   --square_size 0.025
```

これにより、**Bカメラ座標系 → Aカメラ座標系**の変換行列が得られます。

---

## 5. 実行方法（リアルタイム動作）

### 5.1 パネル縦サイズの設定

- `H`：上下LED中心間距離 [m]

例：18cm の場合

```bash
python3 run_overlay.py   --devA /dev/video0   --devB /dev/video2   --H 0.180
```

### 5.2 実行中の挙動

- **Camera B**
  - 上下LEDを検出（緑）
  - 重心を表示（赤）
- **Camera A**
  - 推定されたパネル位置を赤丸でオーバレイ表示
  - 推定距離 Z をテキスト表示

`q` キーで終了します。

---

## 6. 処理フローまとめ

1. カメラBで上下LEDを検出
2. 縦画素距離 `h_px` を算出
3. 単眼距離推定  
   `Z = (f_y * H) / h_px`
4. B画素 → 3D点へ逆投影
5. 外部パラメータで B→A 座標変換
6. カメラA画像へ再投影して描画

---

## 7. 実運用向け改善ポイント

- LED検出ロジックを既存の高精度版に差し替え
- 距離推定値に移動平均 / カルマンフィルタを適用
- `cv2.initUndistortRectifyMap` による歪み補正の高速化
- 同期が必要な場合はタイムスタンプ管理を追加

---

## 8. 注意事項

- 解像度変更時は **必ず再キャリブレーション**
- 広角レンズでは歪み補正が必須
- 遠距離では距離推定誤差が増大するため表示安定化処理を推奨

---

## 9. 想定用途

- 移動ロボット搭載パネルの可視化
- センサ非搭載環境での簡易位置推定
- ロボットUI向けのカメラオーバレイ表示

---

以上で、本システムの **準備から実行までの一連の手順** が確認できます。
