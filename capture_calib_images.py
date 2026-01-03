#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import argparse

def open_cam(dev: str, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {dev}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "pair"], required=True,
                    help="single: capture from one cam, pair: capture synchronized pair")
    ap.add_argument("--devA", default="/dev/video0")
    ap.add_argument("--devB", default="/dev/video2")
    ap.add_argument("--outA", default="calib/A")
    ap.add_argument("--outB", default="calib/B")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    ensure_dir(args.outA)
    ensure_dir(args.outB)

    if args.mode == "single":
        cap = open_cam(args.devA, args.width, args.height, args.fps)
        idx = 0
        print("[single] Keys: s=save, q=quit")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("read failed")
                break
            cv2.imshow("single", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                path = os.path.join(args.outA, f"img_{idx:04d}.png")
                cv2.imwrite(path, frame)
                print("saved", path)
                idx += 1
            elif k == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    else:  # pair
        capA = open_cam(args.devA, args.width, args.height, args.fps)
        capB = open_cam(args.devB, args.width, args.height, args.fps)
        idx = 0
        print("[pair] Keys: s=save pair, q=quit")
        while True:
            okA, fA = capA.read()
            okB, fB = capB.read()
            if not (okA and okB):
                print("read failed")
                break

            vis = cv2.hconcat([fA, fB])
            cv2.imshow("pair (A|B)", vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                pathA = os.path.join(args.outA, f"pair_{idx:04d}_A.png")
                pathB = os.path.join(args.outB, f"pair_{idx:04d}_B.png")
                cv2.imwrite(pathA, fA)
                cv2.imwrite(pathB, fB)
                print("saved", pathA, pathB)
                idx += 1
            elif k == ord('q'):
                break

        capA.release()
        capB.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
