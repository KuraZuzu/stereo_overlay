#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import cv2


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def open_cap(dev, width=None, height=None, fps=None, fourcc=""):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {dev}")
    if fourcc:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        except Exception:
            pass
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    return cap


def next_index(out_dir, regex):
    """
    out_dir内のファイル名から連番の最大値を探し、max+1 を返す。
    regex は 連番部分を (\\d+) でキャプチャする正規表現。
    """
    max_n = 0
    if not os.path.isdir(out_dir):
        return 1
    for name in os.listdir(out_dir):
        m = re.match(regex, name)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return max_n + 1


def run_single(device, out_dir, width=None, height=None, fps=None, fourcc=""):
    ensure_dir(out_dir)
    cap = open_cap(device, width=width, height=height, fps=fps, fourcc=fourcc)
    i = next_index(out_dir, r"single_(\d+)\.png")
    win = "single (s=save, q=quit)"
    shown_size = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if not shown_size:
                h, w = frame.shape[:2]
                print(f"[INFO] actual size: {w}x{h}")
                shown_size = True
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break
            if k == ord('s'):
                path = os.path.join(out_dir, f"single_{i:04d}.png")
                cv2.imwrite(path, frame)
                print("saved:", path)
                i += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_pair(deviceA, deviceB, outA, outB, widthA=None, heightA=None, fpsA=None, fourccA="",
             widthB=None, heightB=None, fpsB=None, fourccB=""):
    ensure_dir(outA)
    ensure_dir(outB)
    capA = open_cap(deviceA, width=widthA, height=heightA, fps=fpsA, fourcc=fourccA)
    capB = open_cap(deviceB, width=widthB, height=heightB, fps=fpsB, fourcc=fourccB)

    iA = next_index(outA, r"pair_(\d+)_A\.png")
    iB = next_index(outB, r"pair_(\d+)_B\.png")
    i = max(iA, iB)  # どちらか進んでいる方に合わせる

    winA = "pair A (s=save, q=quit)"
    winB = "pair B (s=save, q=quit)"
    shown_size = False
    try:
        while True:
            okA, frameA = capA.read()
            okB, frameB = capB.read()
            if okA:
                if not shown_size:
                    hA, wA = frameA.shape[:2]
                    msg = f"[INFO] actual size A: {wA}x{hA}"
                    if okB:
                        hB, wB = frameB.shape[:2]
                        msg += f", B: {wB}x{hB}"
                    print(msg)
                    shown_size = True
                cv2.imshow(winA, frameA)
            if okB:
                cv2.imshow(winB, frameB)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break
            if k == ord('s') and okA and okB:
                pathA = os.path.join(outA, f"pair_{i:04d}_A.png")
                pathB = os.path.join(outB, f"pair_{i:04d}_B.png")
                cv2.imwrite(pathA, frameA)
                cv2.imwrite(pathB, frameB)
                print("saved:", pathA)
                print("saved:", pathB)
                i += 1
    finally:
        capA.release()
        capB.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "pair"], required=True)

    # capture params (optional)
    ap.add_argument("--width", type=int, default=None, help="capture width (single)")
    ap.add_argument("--height", type=int, default=None, help="capture height (single)")
    ap.add_argument("--fps", type=int, default=None, help="capture fps (single)")
    ap.add_argument("--fourcc", default="", help="capture fourcc (single)")

    # single
    ap.add_argument("--device", help="single: /dev/videoX (e.g. /dev/video0)")
    ap.add_argument("--out_dir", help="single: output dir (e.g. calib/A)")

    # pair
    ap.add_argument("--deviceA", help="pair: camera A device (e.g. /dev/video0)")
    ap.add_argument("--deviceB", help="pair: camera B device (e.g. /dev/video2)")
    ap.add_argument("--outA", help="pair: output dir for A (e.g. calib/A)")
    ap.add_argument("--outB", help="pair: output dir for B (e.g. calib/B)")
    ap.add_argument("--widthA", type=int, default=None, help="pair: width for A")
    ap.add_argument("--heightA", type=int, default=None, help="pair: height for A")
    ap.add_argument("--fpsA", type=int, default=None, help="pair: fps for A")
    ap.add_argument("--fourccA", default="", help="pair: fourcc for A")
    ap.add_argument("--widthB", type=int, default=None, help="pair: width for B")
    ap.add_argument("--heightB", type=int, default=None, help="pair: height for B")
    ap.add_argument("--fpsB", type=int, default=None, help="pair: fps for B")
    ap.add_argument("--fourccB", default="", help="pair: fourcc for B")

    args = ap.parse_args()

    print("[press key]\n 's': save the picture\n 'q': quit")

    if args.mode == "single":
        if not args.device or not args.out_dir:
            ap.error("single requires --device and --out_dir")
        run_single(args.device, args.out_dir, width=args.width, height=args.height, fps=args.fps, fourcc=args.fourcc)

    if args.mode == "pair":
        if not (args.deviceA and args.deviceB and args.outA and args.outB):
            ap.error("pair requires --deviceA --deviceB --outA --outB")
        run_pair(
            args.deviceA, args.deviceB, args.outA, args.outB,
            widthA=args.widthA, heightA=args.heightA, fpsA=args.fpsA, fourccA=args.fourccA,
            widthB=args.widthB, heightB=args.heightB, fpsB=args.fpsB, fourccB=args.fourccB,
        )


if __name__ == "__main__":
    main()
