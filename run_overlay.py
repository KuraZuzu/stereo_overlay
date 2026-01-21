#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_overlay.py

- カメラA：表示（設定は基本いじらない）
- カメラB：パネル検出（v4l2制御 + HSV + ペアリング）
- Bで推定したパネル中心（上下LEDの中心）を、外部パラメータでA座標へ変換しA画面にオーバーレイ

注意:
- 内部パラメータ(K/dist)は解像度依存です。実行解像度がキャリブと違う場合、
  PoC向けに K を簡易スケールします（厳密に必要ならその解像度で再キャリブ推奨）。
"""

from __future__ import annotations

import argparse
from typing import Tuple

import cv2
import numpy as np

from panel_detector_b import (
    CamBControls,
    DetectParams,
    apply_camera_controls_b,
    detect_panel_led_pair,
    draw_detection_debug,
)


# ----------------------------
# YAML I/O
# ----------------------------
def _fs_read_mat(fs: cv2.FileStorage, key: str) -> np.ndarray:
    node = fs.getNode(key)
    if node.empty():
        raise RuntimeError(f"YAML missing key: {key}")
    mat = node.mat()
    if mat is None:
        raise RuntimeError(f"YAML key '{key}' is not a matrix")
    return mat


def load_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open intrinsics: {path}")
    try:
        K = _fs_read_mat(fs, "K")
        dist = _fs_read_mat(fs, "dist")
    finally:
        fs.release()
    return K, dist


def load_extrinsics(path: str) -> Tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open extrinsics: {path}")
    try:
        R = _fs_read_mat(fs, "R")
        t = _fs_read_mat(fs, "t")
    finally:
        fs.release()
    return R, t


# ----------------------------
# Camera open
# ----------------------------
def open_cam(dev: str, width: int, height: int, fps: int, *, fourcc: str = "") -> cv2.VideoCapture:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {dev}")

    if fourcc:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        except Exception:
            pass

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    return cap


def parse_size(s: str) -> Tuple[int, int]:
    s2 = s.lower().strip()
    if "x" not in s2:
        raise ValueError("size must be like 640x480")
    w, h = s2.split("x", 1)
    return int(w), int(h)


# ----------------------------
# Intrinsics scaling (PoC用の簡易スケーリング)
# ----------------------------
def scale_K(K: np.ndarray, from_size: Tuple[int, int], to_size: Tuple[int, int]) -> np.ndarray:
    fw, fh = from_size
    tw, th = to_size
    sx = tw / float(fw)
    sy = th / float(fh)
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


# ----------------------------
# Undistort map cache
# ----------------------------
def build_undistorter(K: np.ndarray, dist: np.ndarray, size: Tuple[int, int], *, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, h = size
    newK, _roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)
    return newK, map1, map2


def undistort_with_map(frame: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


# ----------------------------
# Geometry
# ----------------------------
def estimate_depth_from_vertical(top_xy, bottom_xy, fy_px: float, H_m: float):
    h_px = float(np.hypot(top_xy[0] - bottom_xy[0], top_xy[1] - bottom_xy[1]))
    if h_px < 1.0:
        return None, h_px
    Z = (fy_px * H_m) / h_px
    return float(Z), h_px


def backproject_pixel_to_3d(u: float, v: float, Z: float, K: np.ndarray) -> np.ndarray:
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    x = (u - cx) / fx
    y = (v - cy) / fy
    return np.array([[Z * x], [Z * y], [Z]], dtype=np.float64)


def project_3d_to_pixel(P: np.ndarray, K: np.ndarray):
    X, Y, Z = float(P[0, 0]), float(P[1, 0]), float(P[2, 0])
    if Z <= 1e-6:
        return None
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return (float(u), float(v))


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--devA", default="/dev/video0")
    ap.add_argument("--devB", default="/dev/video2")

    ap.add_argument("--intrA", default="calib/intrinsics_A.yaml")
    ap.add_argument("--intrB", default="calib/intrinsics_B.yaml")
    ap.add_argument("--extr", default="calib/extrinsics_B_to_A.yaml")

    # カメラA（表示）
    ap.add_argument("--widthA", type=int, default=640)
    ap.add_argument("--heightA", type=int, default=480)
    ap.add_argument("--fpsA", type=int, default=30)

    # カメラB（検出）
    ap.add_argument("--widthB", type=int, default=640)
    ap.add_argument("--heightB", type=int, default=480)
    ap.add_argument("--fpsB", type=int, default=90)
    ap.add_argument("--fourccB", default="MJPG")

    # キャリブ時の解像度（違う場合は簡易スケール）
    ap.add_argument("--calibA", default="640x480")
    ap.add_argument("--calibB", default="640x480")

    # undistort
    ap.add_argument("--alpha", type=float, default=1.0, help="undistort alpha (1.0=wide, 0.0=crop/zoom)")
    ap.add_argument("--no_undistortA", action="store_true")
    ap.add_argument("--no_undistortB", action="store_true")

    # Bのv4l2設定を切りたい場合用
    ap.add_argument("--no_bctrl", action="store_true", help="do not apply v4l2 controls to camera B")

    ap.add_argument("--H", type=float, required=True, help="vertical LED center distance in meters")

    args = ap.parse_args()

    K_A, dist_A = load_intrinsics(args.intrA)
    K_B, dist_B = load_intrinsics(args.intrB)
    R_BA, t_BA = load_extrinsics(args.extr)  # P_A = R_BA * P_B + t_BA

    capA = open_cam(args.devA, args.widthA, args.heightA, args.fpsA)
    capB = open_cam(args.devB, args.widthB, args.heightB, args.fpsB, fourcc=args.fourccB)

    # Bにだけv4l2設定を適用
    if not args.no_bctrl:
        apply_camera_controls_b(args.devB, capB, CamBControls())

    # 実フレームを1枚読んでサイズ確定（要求が通ってないケースを見える化）
    okA, frameA0 = capA.read()
    okB, frameB0 = capB.read()
    if not (okA and okB):
        raise RuntimeError("initial read failed")

    hA, wA = frameA0.shape[:2]
    hB, wB = frameB0.shape[:2]
    print(f"[INFO] A frame size: {wA}x{hA}  (requested {args.widthA}x{args.heightA})")
    print(f"[INFO] B frame size: {wB}x{hB}  (requested {args.widthB}x{args.heightB})")

    # キャリブ解像度→実行解像度が違う場合の簡易スケール
    calibAw, calibAh = parse_size(args.calibA)
    calibBw, calibBh = parse_size(args.calibB)

    if (calibAw, calibAh) != (wA, hA):
        K_A = scale_K(K_A, (calibAw, calibAh), (wA, hA))
        print(f"[WARN] Scaled K_A from {calibAw}x{calibAh} to {wA}x{hA} (PoC approximation)")
    if (calibBw, calibBh) != (wB, hB):
        K_B = scale_K(K_B, (calibBw, calibBh), (wB, hB))
        print(f"[WARN] Scaled K_B from {calibBw}x{calibBh} to {wB}x{hB} (PoC approximation)")

    # undistort maps
    if args.no_undistortA:
        newK_A, map1A, map2A = K_A, None, None
    else:
        newK_A, map1A, map2A = build_undistorter(K_A, dist_A, (wA, hA), alpha=args.alpha)

    if args.no_undistortB:
        newK_B, map1B, map2B = K_B, None, None
    else:
        newK_B, map1B, map2B = build_undistorter(K_B, dist_B, (wB, hB), alpha=args.alpha)

    params = DetectParams()

    print("Keys: q=quit")

    # ループ（厳密同期不要の前提）
    while True:
        okA, frameA = capA.read()
        okB, frameB = capB.read()
        if not (okA and okB):
            # 一時的な落ちを想定して継続（抜けるよりマシ）
            continue

        if args.no_undistortA:
            undA = frameA
        else:
            undA = undistort_with_map(frameA, map1A, map2A)

        if args.no_undistortB:
            undB = frameB
        else:
            undB = undistort_with_map(frameB, map1B, map2B)

        det = detect_panel_led_pair(undB, params=params)

        overlay_text = "no detection"
        if det is not None:
            # デバッグ描画（B）
            draw_detection_debug(undB, det)

            top = det.top_center
            bottom = det.bottom_center

            # 2点の中点を「パネル中心」として扱う
            uc = 0.5 * (top[0] + bottom[0])
            vc = 0.5 * (top[1] + bottom[1])

            fy_px = float(newK_B[1, 1])
            Z, hpx = estimate_depth_from_vertical(top, bottom, fy_px, args.H)

            if Z is not None:
                P_B = backproject_pixel_to_3d(uc, vc, Z, newK_B)
                P_A = R_BA @ P_B + t_BA
                uvA = project_3d_to_pixel(P_A, newK_A)

                overlay_text = f"Z={Z:.3f}m  hpx={hpx:.1f}  color={det.color}"

                if uvA is not None:
                    uA, vA = uvA
                    if 0 <= uA < undA.shape[1] and 0 <= vA < undA.shape[0]:
                        cv2.circle(undA, (int(uA), int(vA)), 10, (0, 0, 255), 3)
                        cv2.putText(
                            undA,
                            "panel",
                            (int(uA) + 12, int(vA) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

        cv2.putText(
            undA,
            overlay_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Camera A (display + overlay)", undA)
        cv2.imshow("Camera B (detection debug)", undB)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    capA.release()
    capB.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
