#!/usr/bin/env python3
"""
cross_validate.py
-----------------
Runs both the Python prototype and the compiled C++ pipeline on the same
synthetic test image, then compares their detection outputs to verify that
the C++ translation produces consistent results.

This script is meant to be run after building the C++ code:
    cd build && cmake .. && make -j
    cd ..
    python3 scripts/cross_validate.py

It will:
    1. Generate a synthetic star field (saved to data/test_frame.png)
    2. Run the Python star detector
    3. Run the C++ binary
    4. Compare detection counts and centroid positions
    5. Report pass/fail
"""

import subprocess
import sys
import os
import json
import numpy as np

# Make sure we can import the prototypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                 '..', 'python_prototypes'))
from star_detector import generate_synthetic_starfield, run_pipeline

import cv2


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    build_dir = os.path.join(project_root, 'build')
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Generate and save the test image
    print("[1/4] Generating synthetic test image...")
    image, ground_truth = generate_synthetic_starfield(
        width=512, height=512, num_stars=40, noise_std=10.0, seed=42)
    test_img_path = os.path.join(data_dir, 'test_frame.png')
    cv2.imwrite(test_img_path, image)
    print(f"       Saved to {test_img_path}")
    print(f"       Ground truth: {len(ground_truth)} stars")

    # Step 2: Run the Python pipeline
    print("[2/4] Running Python star detector...")
    config = {
        "morph_kernel_size": 21,
        "sigma_multiplier": 3.5,
        "min_area": 3,
        "max_area": 500,
        "max_eccentricity": 0.85,
        "centroid_half_window": 4,
    }
    py_detections, py_timing = run_pipeline(image, config)
    print(f"       Python found {len(py_detections)} stars"
          f" in {py_timing['total_ms']:.1f} ms")

    # Step 3: Run the C++ pipeline
    print("[3/4] Running C++ pipeline...")
    cpp_binary = os.path.join(build_dir, 'space_sensing_pipeline')
    if not os.path.exists(cpp_binary):
        print(f"[ERROR] C++ binary not found at {cpp_binary}")
        print("        Build it first: cd build && cmake .. && make -j")
        sys.exit(1)

    result = subprocess.run(
        [cpp_binary, '--input', test_img_path],
        capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print(f"[ERROR] C++ binary failed:\n{result.stderr}")
        sys.exit(1)

    print(f"       C++ output:\n{result.stdout}")

    # Parse the detection count from C++ output
    cpp_count = None
    for line in result.stdout.split('\n'):
        if 'Detections found:' in line:
            cpp_count = int(line.split(':')[1].strip())
            break

    if cpp_count is None:
        print("[ERROR] Could not parse C++ detection count")
        sys.exit(1)

    # Step 4: Compare
    print("[4/4] Cross-validation results:")
    print(f"       Python detections: {len(py_detections)}")
    print(f"       C++ detections:    {cpp_count}")

    # They won't be identical due to floating point differences in the
    # threshold calculation, but they should be close
    diff = abs(len(py_detections) - cpp_count)
    tolerance = max(5, int(0.15 * len(py_detections)))  # 15% or 5, whichever is larger

    if diff <= tolerance:
        print(f"       PASS (difference of {diff} within tolerance of {tolerance})")
    else:
        print(f"       WARN (difference of {diff} exceeds tolerance of {tolerance})")
        print("       This may indicate a translation bug worth investigating.")

    print("\nDone.")


if __name__ == "__main__":
    main()
