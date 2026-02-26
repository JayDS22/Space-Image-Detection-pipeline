#!/usr/bin/env python3
"""
star_detector.py

Reference implementation of the star/point-source detection pipeline.
This is the version we validate against before porting to C++.

Stages: morph opening for bg -> adaptive threshold -> connected components
with size/shape filtering -> intensity-weighted centroid refinement.

Author: (your name)
Date:   2026-01-15
"""

import numpy as np
import cv2
import argparse
import time
import json
import os


def estimate_background(image, kernel_size=21):
    """Morph opening to knock out anything smaller than the SE, leaving
    just the slowly-varying illumination pattern."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return background


def adaptive_threshold(image, background, sigma_multiplier=3.5):
    """Threshold at mean + k*sigma of the residual noise floor. More
    robust than a hard threshold when gain/exposure changes between frames."""
    residual = cv2.subtract(image, background)

    # only use the bottom 90% of pixels for noise stats (avoid star contamination)
    flat = residual.flatten()
    cutoff = np.percentile(flat, 90)
    noise_pixels = flat[flat <= cutoff]

    mu = np.mean(noise_pixels)
    sigma = np.std(noise_pixels)

    threshold_val = mu + sigma_multiplier * sigma
    threshold_val = max(threshold_val, 1.0)  # safety floor

    _, binary = cv2.threshold(residual, threshold_val, 255, cv2.THRESH_BINARY)
    return binary.astype(np.uint8), residual


def filter_components(binary, min_area=3, max_area=500, max_eccentricity=0.85):
    """CC labeling with size + shape filtering. Stars should be small
    round blobs; elongated things are probably cosmic ray hits."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    detections = []
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # rough eccentricity from bbox aspect ratio
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 1.0
        if (1.0 - aspect) > max_eccentricity:
            continue

        detections.append({
            "label": i,
            "cx": float(centroids[i][0]),
            "cy": float(centroids[i][1]),
            "area": int(area),
            "bbox_w": int(w),
            "bbox_h": int(h),
        })

    return detections, labels


def refine_centroids(residual, detections, labels, half_window=4):
    """Intensity-weighted moment centroid over a small window. Standard
    astrometry trick -- gets us to ~0.1-0.3 px on well-sampled PSFs."""
    rows, cols = residual.shape
    refined = []

    for det in detections:
        cx_init = det["cx"]
        cy_init = det["cy"]

        # Clamp the ROI to image boundaries
        x0 = max(int(cx_init) - half_window, 0)
        x1 = min(int(cx_init) + half_window + 1, cols)
        y0 = max(int(cy_init) - half_window, 0)
        y1 = min(int(cy_init) + half_window + 1, rows)

        roi = residual[y0:y1, x0:x1].astype(np.float64)
        total = roi.sum()
        if total < 1e-6:
            refined.append(det)
            continue

        # Weighted centroid within the ROI
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cx_ref = np.sum(xx * roi) / total
        cy_ref = np.sum(yy * roi) / total

        det["cx_refined"] = float(cx_ref)
        det["cy_refined"] = float(cy_ref)
        refined.append(det)

    return refined


def generate_synthetic_starfield(width=1024, height=1024, num_stars=80,
                                  noise_std=12.0, seed=42):
    """Fake star field with Gaussian PSFs on a gradient background + noise.
    Good enough for pipeline validation when we don't have real frames."""
    rng = np.random.RandomState(seed)
    # Smooth background gradient to simulate uneven illumination
    bg = np.zeros((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            bg[y, x] = 30 + 20 * np.sin(2 * np.pi * x / width) \
                           + 15 * np.cos(2 * np.pi * y / height)

    image = bg.copy()

    # Plant Gaussian point sources at random locations
    star_params = []
    for _ in range(num_stars):
        sx = rng.uniform(20, width - 20)
        sy = rng.uniform(20, height - 20)
        brightness = rng.uniform(60, 220)
        spread = rng.uniform(1.2, 2.5)  # sigma in pixels

        star_params.append((sx, sy, brightness, spread))

        # Stamp the Gaussian into the image
        yy, xx = np.mgrid[max(0, int(sy)-8):min(height, int(sy)+9),
                          max(0, int(sx)-8):min(width, int(sx)+9)]
        gauss = brightness * np.exp(-((xx - sx)**2 + (yy - sy)**2)
                                     / (2 * spread**2))
        image[yy, xx] += gauss

    # Add sensor read noise
    noise = rng.normal(0, noise_std, (height, width))
    image += noise
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image, star_params


def run_pipeline(image, config):
    """Run all four stages on one frame, return detections + per-stage timing."""
    t0 = time.perf_counter()

    background = estimate_background(image, config["morph_kernel_size"])
    t1 = time.perf_counter()

    binary, residual = adaptive_threshold(image, background,
                                          config["sigma_multiplier"])
    t2 = time.perf_counter()

    detections, labels = filter_components(binary,
                                           config["min_area"],
                                           config["max_area"],
                                           config["max_eccentricity"])
    t3 = time.perf_counter()

    refined = refine_centroids(residual, detections, labels,
                               config["centroid_half_window"])
    t4 = time.perf_counter()

    timing = {
        "background_ms": (t1 - t0) * 1000,
        "threshold_ms": (t2 - t1) * 1000,
        "component_filter_ms": (t3 - t2) * 1000,
        "centroid_refine_ms": (t4 - t3) * 1000,
        "total_ms": (t4 - t0) * 1000,
    }

    return refined, timing


def main():
    parser = argparse.ArgumentParser(
        description="Star/point-source detection prototype"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input grayscale image (PNG/TIFF)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a generated synthetic star field instead")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write detections to a JSON file")
    parser.add_argument("--output-image", type=str, default=None,
                        help="Save annotated result image")
    parser.add_argument("--sigma", type=float, default=3.5,
                        help="Sigma multiplier for adaptive threshold")
    parser.add_argument("--morph-kernel", type=int, default=21,
                        help="Morphological kernel diameter")
    args = parser.parse_args()

    config = {
        "morph_kernel_size": args.morph_kernel,
        "sigma_multiplier": args.sigma,
        "min_area": 3,
        "max_area": 500,
        "max_eccentricity": 0.85,
        "centroid_half_window": 4,
    }

    if args.synthetic or args.input is None:
        print("[INFO] Generating synthetic 1024x1024 star field (80 stars)...")
        image, ground_truth = generate_synthetic_starfield()
        print(f"[INFO] Ground truth: {len(ground_truth)} stars planted")
    else:
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[ERROR] Could not read image: {args.input}")
            return
        ground_truth = None

    detections, timing = run_pipeline(image, config)

    print(f"\n--- Detection Results ---")
    print(f"Detections found: {len(detections)}")
    for k, v in timing.items():
        print(f"  {k}: {v:.2f} ms")

    if ground_truth is not None:
        # Simple matching: count how many ground truth stars have a
        # detection within 3 pixels
        matched = 0
        for (sx, sy, _, _) in ground_truth:
            for d in detections:
                rx = d.get("cx_refined", d["cx"])
                ry = d.get("cy_refined", d["cy"])
                if np.hypot(rx - sx, ry - sy) < 3.0:
                    matched += 1
                    break
        recall = matched / len(ground_truth) if ground_truth else 0
        print(f"  Recall vs ground truth: {matched}/{len(ground_truth)}"
              f" ({recall*100:.1f}%)")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"config": config, "detections": detections,
                        "timing": timing}, f, indent=2)
        print(f"[INFO] Detections written to {args.output_json}")

    if args.output_image:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for d in detections:
            cx = d.get("cx_refined", d["cx"])
            cy = d.get("cy_refined", d["cy"])
            cv2.circle(vis, (int(cx), int(cy)), 8, (0, 255, 0), 1)
            cv2.circle(vis, (int(cx), int(cy)), 1, (0, 0, 255), -1)
        cv2.imwrite(args.output_image, vis)
        print(f"[INFO] Annotated image saved to {args.output_image}")


if __name__ == "__main__":
    main()
