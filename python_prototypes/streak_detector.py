#!/usr/bin/env python3
"""
streak_detector.py

Detects linear streaks (satellite passes, debris trails) in space
imagery. Uses Canny + probabilistic Hough + a co-linear clustering step
to merge fragmented detections into full streaks. Intended as the
reference impl before porting to C++.

Author: (your name)
Date:   2026-02-01
"""

import numpy as np
import cv2
import argparse
import time
import json
from star_detector import estimate_background, generate_synthetic_starfield


def preprocess_frame(image, morph_kernel_size=21):
    """BG subtraction + bilateral filter. The bilateral is slow but it
    keeps edges sharp, which matters a lot for streak detection."""
    bg = estimate_background(image, morph_kernel_size)
    residual = cv2.subtract(image, bg)
    # Bilateral: edge-preserving smoothing
    smoothed = cv2.bilateralFilter(residual, d=5, sigmaColor=40,
                                    sigmaSpace=40)
    return smoothed, residual


def detect_edges(image, low_thresh=30, high_thresh=90):
    """Canny with thresholds tuned for typical space surveillance SNR."""
    edges = cv2.Canny(image, low_thresh, high_thresh, apertureSize=3)
    return edges


def detect_lines(edges, rho=1, theta_res=np.pi/180,
                 threshold=40, min_length=30, max_gap=10):
    """HoughLinesP (probabilistic variant) -- gives us endpoints directly
    which we need for the clustering step downstream."""
    lines = cv2.HoughLinesP(edges, rho, theta_res, threshold,
                             minLineLength=min_length, maxLineGap=max_gap)
    if lines is None:
        return []

    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.arctan2(y2 - y1, x2 - x1)
        segments.append({
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "length": float(length),
            "angle_rad": float(angle),
        })
    return segments


def cluster_segments(segments, angle_tol_deg=5.0, dist_tol=15.0):
    """Greedy merging of co-linear segments. Two segments belong together
    if their angles are close and the perp distance between midpoints is
    small. DBSCAN would scale better but this is fine for <200 segments."""
    if not segments:
        return []

    angle_tol = np.deg2rad(angle_tol_deg)
    n = len(segments)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [segments[i]]
        visited[i] = True

        mid_i = np.array([(segments[i]["x1"] + segments[i]["x2"]) / 2,
                           (segments[i]["y1"] + segments[i]["y2"]) / 2])
        ang_i = segments[i]["angle_rad"]

        for j in range(i + 1, n):
            if visited[j]:
                continue

            mid_j = np.array([(segments[j]["x1"] + segments[j]["x2"]) / 2,
                               (segments[j]["y1"] + segments[j]["y2"]) / 2])
            ang_j = segments[j]["angle_rad"]

            # Angle difference, handling wraparound
            d_angle = abs(ang_i - ang_j)
            d_angle = min(d_angle, np.pi - d_angle)

            if d_angle > angle_tol:
                continue

            # Perpendicular distance from midpoint j to the line through i
            direction = np.array([np.cos(ang_i), np.sin(ang_i)])
            diff = mid_j - mid_i
            perp_dist = abs(diff[0] * direction[1] - diff[1] * direction[0])

            if perp_dist <= dist_tol:
                cluster.append(segments[j])
                visited[j] = True

        clusters.append(cluster)

    # For each cluster, compute the merged streak parameters
    merged = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        # Collect all endpoints
        pts = []
        for seg in cluster:
            pts.append([seg["x1"], seg["y1"]])
            pts.append([seg["x2"], seg["y2"]])
        pts = np.array(pts)

        # Fit a line through all points (PCA-style: pick the major axis)
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, 1]  # largest eigenvalue is last

        # Project all points onto the major axis to find endpoints
        projections = centered @ major
        idx_min = np.argmin(projections)
        idx_max = np.argmax(projections)

        merged.append({
            "x1": int(pts[idx_min, 0]),
            "y1": int(pts[idx_min, 1]),
            "x2": int(pts[idx_max, 0]),
            "y2": int(pts[idx_max, 1]),
            "length": float(np.hypot(pts[idx_max, 0] - pts[idx_min, 0],
                                      pts[idx_max, 1] - pts[idx_min, 1])),
            "num_fragments": len(cluster),
        })
    return merged


def add_synthetic_streaks(image, num_streaks=3, seed=123):
    """Draw random linear streaks for testing."""
    rng = np.random.RandomState(seed)
    h, w = image.shape
    result = image.copy().astype(np.float64)
    streak_params = []

    for _ in range(num_streaks):
        x1 = rng.randint(10, w - 10)
        y1 = rng.randint(10, h - 10)
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(80, 300)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        brightness = rng.uniform(100, 200)

        # Clamp to image bounds
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)

        streak_params.append((x1, y1, x2, y2, brightness))

        # Draw with anti-aliasing for realism
        cv2.line(result, (x1, y1), (x2, y2), brightness, thickness=2,
                 lineType=cv2.LINE_AA)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result, streak_params


def run_streak_pipeline(image, config):
    """Full streak detection pipeline on a single frame."""
    t0 = time.perf_counter()

    smoothed, residual = preprocess_frame(image, config["morph_kernel_size"])
    t1 = time.perf_counter()

    edges = detect_edges(smoothed, config["canny_low"], config["canny_high"])
    t2 = time.perf_counter()

    segments = detect_lines(edges,
                            threshold=config["hough_threshold"],
                            min_length=config["hough_min_length"],
                            max_gap=config["hough_max_gap"])
    t3 = time.perf_counter()

    merged = cluster_segments(segments,
                              config["cluster_angle_tol"],
                              config["cluster_dist_tol"])
    t4 = time.perf_counter()

    timing = {
        "preprocess_ms": (t1 - t0) * 1000,
        "edge_detect_ms": (t2 - t1) * 1000,
        "hough_ms": (t3 - t2) * 1000,
        "clustering_ms": (t4 - t3) * 1000,
        "total_ms": (t4 - t0) * 1000,
    }

    return merged, segments, timing


def main():
    parser = argparse.ArgumentParser(
        description="Streak/satellite trail detection prototype"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input grayscale image")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic starfield with streaks")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-image", type=str, default=None)
    args = parser.parse_args()

    config = {
        "morph_kernel_size": 21,
        "canny_low": 30,
        "canny_high": 90,
        "hough_threshold": 40,
        "hough_min_length": 30,
        "hough_max_gap": 10,
        "cluster_angle_tol": 5.0,
        "cluster_dist_tol": 15.0,
    }

    if args.synthetic or args.input is None:
        print("[INFO] Generating synthetic scene with stars and streaks...")
        base_image, _ = generate_synthetic_starfield(num_stars=50, seed=99)
        image, streak_gt = add_synthetic_streaks(base_image, num_streaks=4)
        print(f"[INFO] Ground truth: {len(streak_gt)} streaks inserted")
    else:
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[ERROR] Could not read image: {args.input}")
            return
        streak_gt = None

    merged, raw_segments, timing = run_streak_pipeline(image, config)

    print(f"\n--- Streak Detection Results ---")
    print(f"Raw Hough segments: {len(raw_segments)}")
    print(f"Merged streaks:     {len(merged)}")
    for k, v in timing.items():
        print(f"  {k}: {v:.2f} ms")
    for i, s in enumerate(merged):
        print(f"  Streak {i}: ({s['x1']},{s['y1']})->({s['x2']},{s['y2']})"
              f"  len={s['length']:.1f}px  fragments={s['num_fragments']}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"config": config, "streaks": merged,
                        "timing": timing}, f, indent=2)

    if args.output_image:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Draw raw segments in dim blue
        for seg in raw_segments:
            cv2.line(vis, (seg["x1"], seg["y1"]),
                     (seg["x2"], seg["y2"]), (180, 100, 50), 1)
        # Draw merged streaks in bright green
        for s in merged:
            cv2.line(vis, (s["x1"], s["y1"]),
                     (s["x2"], s["y2"]), (0, 255, 0), 2)
        cv2.imwrite(args.output_image, vis)
        print(f"[INFO] Annotated image saved to {args.output_image}")


if __name__ == "__main__":
    main()
