#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import argparse
import numpy as np
import cv2

def save_intrinsics(path, K, dist, image_size, rms):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("dist", dist)
    fs.write("width", int(image_size[0]))
    fs.write("height", int(image_size[1]))
    fs.write("rms", float(rms))
    fs.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--board_cols", type=int, required=True, help="chessboard inner corners cols")
    ap.add_argument("--board_rows", type=int, required=True, help="chessboard inner corners rows")
    ap.add_argument("--square_size", type=float, required=True, help="square size in meters")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    pattern = os.path.join(args.img_dir, "*.png")
    paths = sorted(glob.glob(pattern))
    if len(paths) < 10:
        raise RuntimeError(f"Need more images (>=10). found={len(paths)}")

    board_size = (args.board_cols, args.board_rows)

    objp = np.zeros((args.board_rows * args.board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.board_cols, 0:args.board_rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []
    imgpoints = []

    image_size = None

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, board_size,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                                                   | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners2)

        if args.show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, board_size, corners2, found)
            cv2.imshow("corners", vis)
            cv2.waitKey(50)

    if args.show:
        cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise RuntimeError(f"Not enough valid detections. valid={len(objpoints)}")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    print("RMS:", rms)
    print("K:\n", K)
    print("dist:\n", dist.ravel())

    save_intrinsics(args.out, K, dist, image_size, rms)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
