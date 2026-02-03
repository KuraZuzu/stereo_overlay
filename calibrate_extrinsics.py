#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import argparse
import numpy as np
import cv2


def load_intrinsics(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open intrinsics file: {path}")
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    width = int(fs.getNode("width").real())
    height = int(fs.getNode("height").real())
    fs.release()
    return K, dist, (width, height)


def save_extrinsics(path, R, t, rms=0.0, note=""):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("R", R)
    fs.write("t", t)
    fs.write("rms", float(rms))
    if note:
        fs.write("note", note)
    fs.release()


def rot_x(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float64)


def rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


def make_ideal_extrinsics(baseline_m, baseline_sign, roll_deg, pitch_deg, yaw_deg):
    """
    Ideal model for B->A transform:
      - Camera coordinates follow OpenCV convention:
          x: right, y: down, z: forward
      - A is mounted above B by 'baseline_m' meters (physical).
      - If axes are perfectly aligned (ideal), R = I.
      - Translation for mapping P_B -> P_A:
            P_A = R * P_B + t
        With A above B, t_y should be +baseline_m (because +y is "down").
        (You can flip with baseline_sign=-1 if your geometry is opposite.)
    """
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    # Rotation from B frame to A frame
    # For manual tweaking, we apply yaw->pitch->roll in camera coordinates.
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

    t = np.array([[0.0],
                  [baseline_sign * baseline_m],
                  [0.0]], dtype=np.float64)
    return R, t


def stereo_mode(args):
    KA, distA, sizeA = load_intrinsics(args.intrA)
    KB, distB, sizeB = load_intrinsics(args.intrB)
    if sizeA != sizeB:
        print("WARN: image size differs A vs B. Better to use same resolution for stereo calibration.")
    image_size = sizeA

    pathsA = sorted(glob.glob(os.path.join(args.dirA, "pair_*_A.png")))
    if len(pathsA) == 0:
        raise RuntimeError("No pair_*_A.png found in dirA (stereo mode requires pair images).")

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

    result = cv2.stereoCalibrate(
        objpoints, imgpointsA, imgpointsB,
        KA, distA, KB, distB, image_size,
        criteria=criteria, flags=flags
    )

    # OpenCV Python bindings can return 5 or 9 values depending on version.
    # 5 values: (rms, R, t, E, F)
    # 9 values: (rms, K1, dist1, K2, dist2, R, t, E, F)
    if len(result) == 5:
        rms, R, t, E, F = result
    elif len(result) == 9:
        rms, _K1, _d1, _K2, _d2, R, t, E, F = result
    else:
        raise RuntimeError(f"Unexpected stereoCalibrate return length: {len(result)}")

    # OpenCV stereoCalibrate returns R, t such that:
    #   points_B = R * points_A + t
    # We want B->A: P_A = R_BA * P_B + t_BA (invert A->B).
    R_AtoB = R
    t_AtoB = t
    R_BtoA = R_AtoB.T
    t_BtoA = -R_AtoB.T @ t_AtoB

    save_extrinsics(args.out, R_BtoA, t_BtoA, rms=rms, note="stereoCalibrate (pair images)")
    print("saved B->A extrinsics:", args.out)


def ideal_mode(args):
    if args.baseline is None:
        raise RuntimeError("--baseline is required in ideal mode.")
    baseline_sign = +1 if args.baseline_sign >= 0 else -1

    R, t = make_ideal_extrinsics(
        baseline_m=args.baseline,
        baseline_sign=baseline_sign,
        roll_deg=args.roll,
        pitch_deg=args.pitch,
        yaw_deg=args.yaw
    )

    note = (
        "IDEAL model (no pair images). "
        "Assume axes aligned, OpenCV camera coords (x right, y down, z forward). "
        f"t_y = baseline_sign({baseline_sign}) * baseline({args.baseline} m). "
        "Optional roll/pitch/yaw are manual tweaks."
    )
    save_extrinsics(args.out, R, t, rms=0.0, note=note)
    print("saved IDEAL B->A extrinsics:", args.out)
    print("R:\n", R)
    print("t:\n", t.ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ideal", "stereo"], default="ideal",
                    help="ideal: generate extrinsics from baseline only. stereo: use pair images (stereoCalibrate).")

    # Output (both modes)
    ap.add_argument("--out", required=True)

    # Ideal-mode params
    ap.add_argument("--baseline", type=float, default=None,
                    help="Vertical distance between camera centers [m]. A is above B by default.")
    ap.add_argument("--baseline_sign", type=int, default=+1,
                    help="Sign for baseline translation t_y. +1: A above B (default). -1: flip if overlay goes opposite.")
    ap.add_argument("--roll", type=float, default=0.0, help="Manual tweak roll [deg] (around x).")
    ap.add_argument("--pitch", type=float, default=0.0, help="Manual tweak pitch [deg] (around y).")
    ap.add_argument("--yaw", type=float, default=0.0, help="Manual tweak yaw [deg] (around z).")

    # Stereo-mode params (only needed if --mode stereo)
    ap.add_argument("--dirA", default="calib/A")
    ap.add_argument("--dirB", default="calib/B")
    ap.add_argument("--intrA", default="calib/intrinsics_A.yaml")
    ap.add_argument("--intrB", default="calib/intrinsics_B.yaml")
    ap.add_argument("--board_cols", type=int, default=9)
    ap.add_argument("--board_rows", type=int, default=6)
    ap.add_argument("--square_size", type=float, default=0.025)
    ap.add_argument("--show", action="store_true")

    args = ap.parse_args()

    if args.mode == "ideal":
        ideal_mode(args)
    else:
        stereo_mode(args)


if __name__ == "__main__":
    main()
