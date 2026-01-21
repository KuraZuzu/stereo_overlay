#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""panel_detector_b.py

カメラB（検出用）のための小さなモジュール。

- v4l2-ctl でカメラBの露光/ゲイン/ガンマ/フォーカス等を起動時に固定（可能な範囲で）
- HSV + 矩形抽出 + 上下ペアリング（同色LEDバー）でパネル検出
  - 複数候補がある場合は「画面中心xに最も近いもの」を返す

run_overlay.py から import して使う想定です。
"""

from __future__ import annotations

import re
import time
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import cv2


# =========================
# カメラB: 初期設定（必要なら run_overlay 側で上書きしてもOK）
# =========================
@dataclass(frozen=True)
class CamBControls:
    # --- User Controls ---
    brightness: int = 0                  # [-64..64]
    contrast: int = 32                   # [0..??]  ※機種依存
    saturation: int = 90                # [0..??]  ※機種依存
    hue: int = 0                         # [-??..??]※機種依存
    white_balance_automatic: int = 0     # [0/1]
    gamma: int = 110                     # [??..??] ※機種依存
    gain: int = 0                        # [0..??]  ※機種依存
    power_line_frequency: int = 1        # 0=Off,1=50Hz,2=60Hz
    white_balance_temperature: int = 4600
    sharpness: int = 3
    backlight_compensation: int = 0     # ※機種依存

    # --- Camera Controls ---
    auto_exposure: int = 1               # 1=Manual, 3=Auto
    exposure_time_absolute: int = 4      # ※機種依存（だいたい 100µs 単位が多いらしい）

    focus_automatic_continuous: int = 0
    focus_absolute: int = 0

    # --- Optional ---
    # pan_absolute: int = 0
    # tilt_absolute: int = 0
    # zoom_absolute: int = 0


# =========================
# 検出パラメータ（既存ロジック）
# =========================
@dataclass(frozen=True)
class DetectParams:
    kernel_sz: int = 3
    width_tol: float = 0.6
    min_h_overlap: float = 0.05
    min_v_gap: int = 1
    min_box_h: int = 1
    min_box_w: int = 4


# HSVレンジ（必要になったらここだけ調整すればOK）
DEFAULT_HSV_CFG: Dict[str, Dict[str, int]] = {
    "blue":  {"H_low": 100, "H_high": 135, "S_low": 180, "S_high": 255, "V_low": 120, "V_high": 255},
    "red1":  {"H_low":   0, "H_high":  15},
    "red2":  {"H_low": 165, "H_high": 179},
    "redSV": {"S_low": 180, "S_high": 255, "V_low": 120, "V_high": 255},
}


# -------------------------
# v4l2 helpers
# -------------------------
def dev_to_path(dev: Union[int, str]) -> str:
    """v4l2-ctl 用に /dev/videoX 形式へ正規化."""
    if isinstance(dev, int):
        return f"/dev/video{dev}"
    s = str(dev).strip()
    if s.isdigit():
        return f"/dev/video{int(s)}"
    if re.match(r"^/dev/video\d+$", s):
        return s
    return ""


def has_v4l2_ctl() -> bool:
    return shutil.which("v4l2-ctl") is not None


def v4l2_set(dev_path: str, name: str, value) -> bool:
    """v4l2-ctl で設定。失敗しても落とさず False."""
    if (not dev_path) or (not has_v4l2_ctl()):
        return False
    try:
        r = subprocess.run(
            ["v4l2-ctl", "-d", dev_path, f"--set-ctrl={name}={value}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return (r.returncode == 0)
    except Exception:
        return False


def apply_camera_controls_b(
    dev: Union[int, str],
    cap: cv2.VideoCapture,
    ctrl: CamBControls,
    *,
    settle_sec: float = 0.05,
) -> None:
    """カメラBにだけ設定を適用する。

    - 基本は v4l2-ctl（優先）
    - だめなら cap.set を試す（効かない機種も多い）
    """
    dev_path = dev_to_path(dev)

    # オート系は「切ってから値を入れる」順番が重要なことが多い
    v4l2_set(dev_path, "auto_exposure", ctrl.auto_exposure)
    v4l2_set(dev_path, "white_balance_automatic", ctrl.white_balance_automatic)
    v4l2_set(dev_path, "focus_automatic_continuous", ctrl.focus_automatic_continuous)

    # 主要項目
    v4l2_set(dev_path, "exposure_time_absolute", ctrl.exposure_time_absolute)
    v4l2_set(dev_path, "gain", ctrl.gain)
    v4l2_set(dev_path, "brightness", ctrl.brightness)
    v4l2_set(dev_path, "contrast", ctrl.contrast)
    v4l2_set(dev_path, "saturation", ctrl.saturation)
    v4l2_set(dev_path, "hue", ctrl.hue)
    v4l2_set(dev_path, "gamma", ctrl.gamma)
    v4l2_set(dev_path, "power_line_frequency", ctrl.power_line_frequency)
    v4l2_set(dev_path, "white_balance_temperature", ctrl.white_balance_temperature)
    v4l2_set(dev_path, "sharpness", ctrl.sharpness)
    v4l2_set(dev_path, "backlight_compensation", ctrl.backlight_compensation)

    # 位置系（必要なら）
    # v4l2_set(dev_path, "pan_absolute", ctrl.pan_absolute)
    # v4l2_set(dev_path, "tilt_absolute", ctrl.tilt_absolute)
    # v4l2_set(dev_path, "zoom_absolute", ctrl.zoom_absolute)

    # focus_absolute は inactive の可能性があるので最後に（失敗しても無視）
    v4l2_set(dev_path, "focus_absolute", ctrl.focus_absolute)

    # OpenCVプロパティのフォールバック（効かない機種が多い）
    try:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(ctrl.brightness))
        cap.set(cv2.CAP_PROP_CONTRAST, float(ctrl.contrast))
        cap.set(cv2.CAP_PROP_SATURATION, float(ctrl.saturation))
        cap.set(cv2.CAP_PROP_GAIN, float(ctrl.gain))
        cap.set(cv2.CAP_PROP_GAMMA, float(ctrl.gamma))
        cap.set(cv2.CAP_PROP_SHARPNESS, float(ctrl.sharpness))
        cap.set(cv2.CAP_PROP_EXPOSURE, float(ctrl.exposure_time_absolute))
        try:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(ctrl.white_balance_temperature))
        except Exception:
            pass
    except Exception:
        pass

    # 反映待ち（機種によっては必要）
    if settle_sec > 0:
        time.sleep(settle_sec)


# -------------------------
# detection helpers
# -------------------------
def _get_led_mask(hsv: np.ndarray, color: str, hsv_cfg: Dict[str, Dict[str, int]]) -> Optional[np.ndarray]:
    if color == "blue":
        c = hsv_cfg["blue"]
        lo = np.array([c["H_low"], c["S_low"], c["V_low"]], dtype=np.uint8)
        hi = np.array([c["H_high"], c["S_high"], c["V_high"]], dtype=np.uint8)
        return cv2.inRange(hsv, lo, hi)

    if color == "red":
        r1 = hsv_cfg["red1"]
        r2 = hsv_cfg["red2"]
        rsv = hsv_cfg["redSV"]
        lo1 = np.array([r1["H_low"], rsv["S_low"], rsv["V_low"]], dtype=np.uint8)
        hi1 = np.array([r1["H_high"], rsv["S_high"], rsv["V_high"]], dtype=np.uint8)
        lo2 = np.array([r2["H_low"], rsv["S_low"], rsv["V_low"]], dtype=np.uint8)
        hi2 = np.array([r2["H_high"], rsv["S_high"], rsv["V_high"]], dtype=np.uint8)
        m1 = cv2.inRange(hsv, lo1, hi1)
        m2 = cv2.inRange(hsv, lo2, hi2)
        return cv2.bitwise_or(m1, m2)

    return None


def _find_boxes(mask: np.ndarray, p: DetectParams) -> List[Tuple[int, int, int, int]]:
    kernel = np.ones((p.kernel_sz, p.kernel_sz), np.uint8)
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (h < p.min_box_h) or (w < p.min_box_w):
            continue
        boxes.append((x, y, w, h))
    return boxes


def _horiz_overlap_ratio(b1, b2) -> float:
    x1, _, w1, _ = b1
    x2, _, w2, _ = b2
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    overlap = max(0, right - left)
    denom = float(min(w1, w2))
    return 0.0 if denom <= 0 else overlap / denom


def _pair_boxes_same_color(
    boxes: List[Tuple[int, int, int, int]],
    p: DetectParams,
) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    boxes_sorted = sorted(boxes, key=lambda b: b[1])  # y昇順
    paired: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]] = []
    used_bottom = set()

    for i, top in enumerate(boxes_sorted):
        x_t, y_t, w_t, h_t = top
        best_j = -1
        best_dy = None

        for j, bottom in enumerate(boxes_sorted):
            if (j == i) or (j in used_bottom):
                continue

            x_b, y_b, w_b, h_b = bottom
            dy = y_b - (y_t + h_t)
            if dy < p.min_v_gap:
                continue
            if abs(w_t - w_b) > p.width_tol * max(w_t, w_b):
                continue
            if _horiz_overlap_ratio(top, bottom) < p.min_h_overlap:
                continue

            if (best_dy is None) or (dy < best_dy):
                best_dy = dy
                best_j = j

        if best_j >= 0:
            paired.append((top, boxes_sorted[best_j]))
            used_bottom.add(best_j)

    return paired


def _bbox_union(top, bottom):
    x1, y1, w1, h1 = top
    x2, y2, w2, h2 = bottom
    x_min = min(x1, x2)
    y_min = y1
    x_max = max(x1 + w1, x2 + w2)
    y_max = y2 + h2
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0
    return (x_min, y_min, w, h), (cx, cy)


# -------------------------
# public API
# -------------------------
@dataclass(frozen=True)
class PanelDetection:
    color: str
    top_box: Tuple[int, int, int, int]
    bottom_box: Tuple[int, int, int, int]
    union_box: Tuple[int, int, int, int]
    top_center: Tuple[float, float]
    bottom_center: Tuple[float, float]
    union_center: Tuple[float, float]


def detect_panel_led_pair(
    frame_bgr: np.ndarray,
    *,
    hsv_cfg: Optional[Dict[str, Dict[str, int]]] = None,
    params: Optional[DetectParams] = None,
) -> Optional[PanelDetection]:
    """上下LEDバー（同色）のペアを 1つ返す。

    - 複数候補がある場合は「画面中心xに最も近いもの」を採用
    - 戻り値の top_center / bottom_center を run_overlay 側の Z 推定に使う想定
    """
    if hsv_cfg is None:
        hsv_cfg = DEFAULT_HSV_CFG
    if params is None:
        params = DetectParams()

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, w = frame_bgr.shape[:2]
    cx0 = w / 2.0

    best: Optional[PanelDetection] = None
    best_dx: Optional[float] = None

    for c in ("blue", "red"):
        mask = _get_led_mask(hsv, c, hsv_cfg)
        if mask is None:
            continue

        boxes = _find_boxes(mask, params)
        pairs = _pair_boxes_same_color(boxes, params)

        for top, bottom in pairs:
            union_box, union_center = _bbox_union(top, bottom)

            # 矩形中心（LEDバー中心の代理）
            tx, ty, tw, th = top
            bx, by, bw, bh = bottom
            top_center = (tx + tw / 2.0, ty + th / 2.0)
            bottom_center = (bx + bw / 2.0, by + bh / 2.0)

            dx = abs(union_center[0] - cx0)
            if (best_dx is None) or (dx < best_dx):
                best_dx = dx
                best = PanelDetection(
                    color=c,
                    top_box=top,
                    bottom_box=bottom,
                    union_box=union_box,
                    top_center=top_center,
                    bottom_center=bottom_center,
                    union_center=union_center,
                )

    return best


def draw_detection_debug(img_bgr: np.ndarray, det: PanelDetection) -> None:
    """検出結果のデバッグ描画（Bウインドウ用）"""
    box_col = (255, 0, 0) if det.color == "blue" else (0, 0, 255)

    for (x, y, w, h) in (det.top_box, det.bottom_box):
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), box_col, 2)

    ux, uy, uw, uh = det.union_box
    cv2.rectangle(img_bgr, (ux, uy), (ux + uw, uy + uh), (0, 255, 0), 2)

    # 中心
    cv2.circle(img_bgr, (int(det.top_center[0]), int(det.top_center[1])), 5, (0, 255, 0), 2)
    cv2.circle(img_bgr, (int(det.bottom_center[0]), int(det.bottom_center[1])), 5, (0, 255, 0), 2)
    cv2.circle(img_bgr, (int(det.union_center[0]), int(det.union_center[1])), 5, (0, 0, 255), 2)
