#!/usr/bin/env python3
"""
calibrate_camera.py — measure the webcam's lens (intrinsics + distortion) with a checkerboard.

Works over SSH with NO window: it watches the camera, and every time it sees the
whole checkerboard in a NEW position it saves that view automatically. You just
move the board around in front of the camera until it has enough views.

    python3 calibrate_camera.py                 # capture 20 views, then calibrate
    python3 calibrate_camera.py --from-dir calib_images   # re-run calibration on saved views

Your board: 6 x 9 squares  ->  5 x 8 INNER corners (where four squares meet).

Output: camera.yaml  (camera_matrix, distortion_coefficients, image_width, image_height)
        -> used by the tracker with  --calib camera.yaml
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cols", type=int, default=5, help="inner corners across")
    p.add_argument("--rows", type=int, default=8, help="inner corners down")
    p.add_argument("--square-mm", type=float, default=25.0, help="size of one square in mm (measure it)")
    p.add_argument("--target", type=int, default=20, help="how many views to collect")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--images-dir", default="calib_images")
    p.add_argument("--from-dir", default=None, help="skip the camera; calibrate from saved images")
    p.add_argument("--out", default="camera.yaml")
    return p.parse_args()


def find_corners(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        return None
    # refine each corner to sub-pixel accuracy
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def describe(corners, w, h):
    """Where the board is (centre, as fraction of image) and how big it looks."""
    pts = corners.reshape(-1, 2)
    cx, cy = pts.mean(axis=0)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    size = ((x1 - x0) * (y1 - y0)) / (w * h)          # fraction of the image the board covers
    # tilt cue: ratio of top edge length to bottom edge length (perspective)
    return np.array([cx / w, cy / h, size])


def is_new_view(desc, saved):
    """Only keep views that differ from every saved one (position or size)."""
    for d in saved:
        moved = np.hypot(desc[0] - d[0], desc[1] - d[1])
        resized = abs(desc[2] - d[2]) / max(d[2], 1e-6)
        if moved < 0.08 and resized < 0.20:
            return False
    return True


def region_name(desc):
    col = ["left", "centre", "right"][min(int(desc[0] * 3), 2)]
    row = ["top", "middle", "bottom"][min(int(desc[1] * 3), 2)]
    return f"{row}-{col}"


def capture(args, pattern):
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit("Cannot open camera")
    for _ in range(10):                               # let exposure settle
        cap.read()
    ok, frame = cap.read()
    if not ok:
        sys.exit("Camera returned no frames")
    h, w = frame.shape[:2]
    print(f"Camera {w}x{h}. Looking for a {pattern[0]}x{pattern[1]} inner-corner board.")
    print("Move the board: all 9 regions of the image, near AND far, tilted left/right/up/down.")
    print("Hold still ~1 s at each spot. Ctrl+C to stop early.\n")

    os.makedirs(args.images_dir, exist_ok=True)
    saved_desc, regions = [], set()
    last_save, last_msg = 0.0, 0.0
    try:
        while len(saved_desc) < args.target:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, pattern)
            now = time.time()
            if corners is None:
                if now - last_msg > 2.0:
                    print("  ... board not fully visible (all corners must be in view)")
                    last_msg = now
                continue
            desc = describe(corners, w, h)
            if now - last_save < 0.8 or not is_new_view(desc, saved_desc):
                if now - last_msg > 2.0:
                    print("  ... seen, but too similar to a saved view — move/tilt it")
                    last_msg = now
                continue
            n = len(saved_desc) + 1
            cv2.imwrite(os.path.join(args.images_dir, f"view_{n:02d}.png"), frame)
            annotated = frame.copy()
            cv2.drawChessboardCorners(annotated, pattern, corners, True)
            cv2.imwrite(os.path.join(args.images_dir, f"view_{n:02d}_corners.jpg"), annotated)
            saved_desc.append(desc)
            regions.add(region_name(desc))
            missing = 9 - len(regions)
            print(f"  saved {n:2d}/{args.target}  at {region_name(desc):14s} size {desc[2]*100:4.1f}% of image"
                  f"   (regions still empty: {missing})")
            last_save = last_msg = now
    except KeyboardInterrupt:
        print("\nStopped early.")
    cap.release()
    return (w, h)


def calibrate(args, pattern, image_size=None):
    files = sorted(f for f in glob.glob(os.path.join(args.images_dir, "view_*.png")))
    if len(files) < 10:
        sys.exit(f"Only {len(files)} views in {args.images_dir}/ — need at least 10.")

    # 3D corner positions on the flat board (z = 0), in mm
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * args.square_mm

    obj_pts, img_pts, used = [], [], []
    for f in files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        c = find_corners(gray, pattern)
        if c is None:
            print(f"  skip {os.path.basename(f)} (corners not found on re-read)")
            continue
        obj_pts.append(objp)
        img_pts.append(c)
        used.append(f)

    print(f"\nCalibrating from {len(used)} views at {image_size[0]}x{image_size[1]} ...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None)

    # per-view error: how far the model's predicted corners land from the detected ones
    per_view = []
    for i in range(len(used)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        diff = proj.reshape(-1, 2) - img_pts[i].reshape(-1, 2)   # same shape on any OpenCV version
        per_view.append(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hfov = 2 * np.degrees(np.arctan(image_size[0] / (2 * fx)))
    print("\n=== Result ===")
    print(f"Reprojection error (RMS): {rms:.3f} px   (good < 0.5, OK < 1.0)")
    print(f"fx = {fx:.1f}  fy = {fy:.1f}  cx = {cx:.1f}  cy = {cy:.1f}")
    print(f"Horizontal field of view: {hfov:.1f} deg   (tracker's default guess was 70)")
    print(f"Distortion k1,k2,p1,p2,k3: {np.round(dist.ravel(), 4)}")
    worst = np.argsort(per_view)[::-1][:3]
    print("Worst views: " + ", ".join(f"{os.path.basename(used[i])} {per_view[i]:.2f}px" for i in worst))

    fs = cv2.FileStorage(args.out, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", dist)
    fs.write("rms_reprojection_error", float(rms))
    fs.write("square_size_mm", float(args.square_mm))
    fs.release()
    print(f"\nSaved {args.out}  ->  use with:  ./tool_tracker_trt --calib {args.out}")
    return rms, K, dist


def main():
    args = parse_args()
    pattern = (args.cols, args.rows)
    if args.from_dir:
        args.images_dir = args.from_dir
    else:
        capture(args, pattern)
    calibrate(args, pattern)


if __name__ == "__main__":
    main()
