#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import argparse
import numpy as np
import cv2

def load_intrinsics(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    width = int(fs.getNode("width").real())
    height = int(fs.getNode("height").real())
    fs.release()
    return K, dist, (width, height)

def save_extrinsics(path, R, t, rms):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("R", R)
    fs.write("t", t)
    fs.write("rms", float(rms))
    fs.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirA", required=True)
    ap.add_argument("--dirB", required=True)
    ap.add_argument("--intrA", required=True)
    ap.add_argument("--intrB", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--board_cols", type=int, required=True)
    ap.add_argument("--board_rows", type=int, required=True)
    ap.add_argument("--square_size", type=float, required=True)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    KA, distA, sizeA = load_intrinsics(args.intrA)
    KB, distB, sizeB = load_intrinsics(args.intrB)
    if sizeA != sizeB:
        print("WARN: image size differs A vs B. Better to use same resolution for calibration.")
    image_size = sizeA

    # pair filenames assumption
    pathsA = sorted(glob.glob(os.path.join(args.dirA, "pair_*_A.png")))
    if len(pathsA) == 0:
        raise RuntimeError("No pair_*_A.png found in dirA")

    board_size = (args.board_cols, args.board_rows)

    objp = np.zeros((args.board_rows * args.board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.board_cols, 0:args.board_rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []
    imgpointsA = []
    imgpointsB = []

    for pA in pathsA:
        base = os.path.basename(pA).replace("_A.png", "")
        pB = os.path.join(args.dirB, base + "_B.png")
        if not os.path.exists(pB):
            continue

        imgA = cv2.imread(pA)
        imgB = cv2.imread(pB)
        if imgA is None or imgB is None:
            continue

        gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

        fA, cA = cv2.findChessboardCorners(gA, board_size)
        fB, cB = cv2.findChessboardCorners(gB, board_size)
        if not (fA and fB):
            continue

        cA2 = cv2.cornerSubPix(gA, cA, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cB2 = cv2.cornerSubPix(gB, cB, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        objpoints.append(objp)
        imgpointsA.append(cA2)
        imgpointsB.append(cB2)

        if args.show:
            visA = imgA.copy()
            visB = imgB.copy()
            cv2.drawChessboardCorners(visA, board_size, cA2, True)
            cv2.drawChessboardCorners(visB, board_size, cB2, True)
            cv2.imshow("A", visA)
            cv2.imshow("B", visB)
            cv2.waitKey(50)

    if args.show:
        cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise RuntimeError(f"Not enough valid pairs. valid={len(objpoints)}")

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)

    rms, R, t, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsA, imgpointsB,
        KA, distA, KB, distB, image_size,
        criteria=criteria, flags=flags
    )

    print("Stereo RMS:", rms)
    print("R (B->A?) returned R maps object from A to B depending on convention; we will store as R_AB, t_AB from stereoCalibrate output.")
    print("R:\n", R)
    print("t:\n", t.ravel())

    # OpenCV stereoCalibrate returns R, t such that:
    # points_B = R * points_A + t  (common convention in OpenCV examples)
    # We want B->A, so invert.
    R_AtoB = R
    t_AtoB = t
    R_BtoA = R_AtoB.T
    t_BtoA = -R_AtoB.T @ t_AtoB

    save_extrinsics(args.out, R_BtoA, t_BtoA, rms)
    print("saved B->A extrinsics:", args.out)

if __name__ == "__main__":
    main()
