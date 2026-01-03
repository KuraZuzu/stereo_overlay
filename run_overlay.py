#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse

def load_intrinsics(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    fs.release()
    return K, dist

def load_extrinsics(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    R = fs.getNode("R").mat()
    t = fs.getNode("t").mat()
    fs.release()
    return R, t

def open_cam(dev: str, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {dev}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

def undistort(frame, K, dist):
    h, w = frame.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
    und = cv2.undistort(frame, K, dist, None, newK)
    return und, newK

def detect_led_pair(frame_bgr):
    """
    最小例：明るいblobを探して、上(小さいy)と下(大きいy)をLEDとして返す。
    実運用では、あなたの「上下LEDバー（同色）+形状制約」へ差し替え推奨。
    Return:
      (top_xy, bottom_xy) in pixel coords, or None
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # かなり強めの閾値。環境に応じて調整。
    _, binimg = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) < 2:
        return None

    # 面積の大きい順に上位を取る
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"]/M["m00"]
        cy = M["m01"]/M["m00"]
        centers.append((cx, cy))

    if len(centers) < 2:
        return None

    # yでソートして上と下を選択
    centers = sorted(centers, key=lambda p: p[1])
    top = centers[0]
    bottom = centers[-1]
    return top, bottom

def estimate_depth_from_vertical(top, bottom, fy_px, H_m):
    # LED中心間の画素距離（2D）
    h_px = np.hypot(top[0]-bottom[0], top[1]-bottom[1])
    if h_px < 1.0:
        return None, h_px
    Z = (fy_px * H_m) / h_px
    return Z, h_px

def backproject_pixel_to_3d(u, v, Z, K):
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    P = np.array([[Z*x],[Z*y],[Z]], dtype=np.float64)
    return P

def project_3d_to_pixel(P, K):
    X, Y, Z = P[0,0], P[1,0], P[2,0]
    if Z <= 1e-6:
        return None
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    u = fx*(X/Z) + cx
    v = fy*(Y/Z) + cy
    return (float(u), float(v))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devA", default="/dev/video0")
    ap.add_argument("--devB", default="/dev/video2")
    ap.add_argument("--intrA", default="calib/intrinsics_A.yaml")
    ap.add_argument("--intrB", default="calib/intrinsics_B.yaml")
    ap.add_argument("--extr", default="calib/extrinsics_B_to_A.yaml")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--H", type=float, required=True, help="vertical LED center distance in meters")
    args = ap.parse_args()

    K_A, dist_A = load_intrinsics(args.intrA)
    K_B, dist_B = load_intrinsics(args.intrB)
    R_BA, t_BA = load_extrinsics(args.extr)  # maps P_A = R_BA * P_B + t_BA

    capA = open_cam(args.devA, args.width, args.height, args.fps)
    capB = open_cam(args.devB, args.width, args.height, args.fps)

    print("Keys: q=quit")

    while True:
        okA, frameA = capA.read()
        okB, frameB = capB.read()
        if not (okA and okB):
            print("read failed")
            break

        undA, newK_A = undistort(frameA, K_A, dist_A)
        undB, newK_B = undistort(frameB, K_B, dist_B)

        pair = detect_led_pair(undB)

        overlay_text = "no detection"
        if pair is not None:
            top, bottom = pair
            # B重心
            uc = 0.5*(top[0] + bottom[0])
            vc = 0.5*(top[1] + bottom[1])

            fy_px = newK_B[1,1]
            Z, hpx = estimate_depth_from_vertical(top, bottom, fy_px, args.H)

            if Z is not None:
                # B画素→Bカメラ座標3D
                P_B = backproject_pixel_to_3d(uc, vc, Z, newK_B)
                # B→Aへ
                P_A = R_BA @ P_B + t_BA
                # Aへ投影
                uvA = project_3d_to_pixel(P_A, newK_A)

                overlay_text = f"Z={Z:.3f}m  hpx={hpx:.1f}"
                # B側表示（デバッグ）
                cv2.circle(undB, (int(top[0]), int(top[1])), 6, (0,255,0), 2)
                cv2.circle(undB, (int(bottom[0]), int(bottom[1])), 6, (0,255,0), 2)
                cv2.circle(undB, (int(uc), int(vc)), 6, (0,0,255), 2)

                if uvA is not None:
                    uA, vA = uvA
                    if 0 <= uA < undA.shape[1] and 0 <= vA < undA.shape[0]:
                        cv2.circle(undA, (int(uA), int(vA)), 10, (0,0,255), 3)
                        cv2.putText(undA, "panel", (int(uA)+12, int(vA)-12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(undA, overlay_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.imshow("Camera A (display + overlay)", undA)
        cv2.imshow("Camera B (detection debug)", undB)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    capA.release()
    capB.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
