#!/usr/bin/env python3
"""
visualize_pipeline.py

Generates a multi-panel figure showing every stage of the detection pipeline
side by side, plus the final annotated outputs. Meant for the README / docs.

Run from project root:
    python3 scripts/visualize_pipeline.py
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_prototypes'))
from star_detector import (estimate_background, adaptive_threshold,
                           filter_components, refine_centroids,
                           generate_synthetic_starfield)
from streak_detector import (preprocess_frame, detect_edges, detect_lines,
                             cluster_segments, add_synthetic_streaks)


def make_colorbar(height, width=30):
    """Quick vertical gradient strip for intensity reference."""
    bar = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        bar[y, :] = int(255 * (1.0 - y / height))
    return cv2.cvtColor(bar, cv2.COLOR_GRAY2BGR)


def label_image(img, text, pos=(10, 25), scale=0.6, color=(0, 255, 200)):
    """Burn a text label onto the image (top-left by default)."""
    out = img.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                cv2.LINE_AA)
    return out


def to_bgr(gray):
    """Convenience for stacking grayscale panels into a color canvas."""
    if len(gray.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, 'data')
    os.makedirs(out_dir, exist_ok=True)

    # --- generate the test scene ---
    print("Generating synthetic scene...")
    base_img, star_gt = generate_synthetic_starfield(
        width=512, height=512, num_stars=60, noise_std=10.0, seed=42)
    scene, streak_gt = add_synthetic_streaks(base_img, num_streaks=3, seed=77)

    # =========================================================
    #  PANEL 1: Full pipeline stage visualization (2x3 grid)
    # =========================================================
    sz = (512, 512)

    # -- star path --
    bg = estimate_background(scene, 21)
    binary, residual = adaptive_threshold(scene, bg, 3.5)

    # -- streak path --
    smoothed, _ = preprocess_frame(scene, 21)
    edges = detect_edges(smoothed, 30, 90)
    raw_segs = detect_lines(edges, threshold=40, min_length=30, max_gap=10)
    merged_streaks = cluster_segments(raw_segs, 5.0, 15.0)

    # -- star detections --
    dets, labels = filter_components(binary, 3, 500, 0.85)
    dets = refine_centroids(residual, dets, labels, 4)

    # build each panel at 512x512
    p_input   = label_image(to_bgr(scene),    "(a) Input frame")
    p_bg      = label_image(to_bgr(bg),       "(b) Background est.")
    p_resid   = label_image(to_bgr(residual), "(c) Residual (bg subtracted)")

    # threshold binary -- scale up so it's visible
    p_binary  = label_image(to_bgr(binary),   "(d) Adaptive threshold")

    # edges
    p_edges   = label_image(to_bgr(edges),    "(e) Canny edges")

    # final composite
    composite = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    # draw star detections: green circles
    for d in dets:
        cx = d.get("cx_refined", d["cx"])
        cy = d.get("cy_refined", d["cy"])
        cv2.circle(composite, (int(cx), int(cy)), 8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(composite, (int(cx), int(cy)), 1, (0, 180, 255), -1, cv2.LINE_AA)
    # draw raw hough segments thin
    for seg in raw_segs:
        cv2.line(composite, (seg["x1"], seg["y1"]),
                 (seg["x2"], seg["y2"]), (200, 120, 60), 1, cv2.LINE_AA)
    # draw merged streaks thick
    for s in merged_streaks:
        cv2.line(composite, (s["x1"], s["y1"]),
                 (s["x2"], s["y2"]), (80, 80, 255), 2, cv2.LINE_AA)
    p_final = label_image(composite, "(f) Detections overlay")

    # tile into 2 rows x 3 cols
    row1 = np.hstack([p_input, p_bg, p_resid])
    row2 = np.hstack([p_binary, p_edges, p_final])
    pipeline_grid = np.vstack([row1, row2])

    # add a title bar
    title_h = 40
    title_bar = np.zeros((title_h, pipeline_grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, "Pipeline Stage Visualization  |  Stars: green circles  |  Streaks: red lines  |  Hough segments: blue",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    pipeline_grid = np.vstack([title_bar, pipeline_grid])

    pipeline_path = os.path.join(out_dir, 'pipeline_stages.png')
    cv2.imwrite(pipeline_path, pipeline_grid)
    print(f"  -> {pipeline_path}")

    # =========================================================
    #  PANEL 2: Star detection detail (zoomed region)
    # =========================================================
    # pick a 200x200 crop from the center-ish area where stars are likely
    cx0, cy0 = 180, 180
    crop_sz = 200
    crop_scene = scene[cy0:cy0+crop_sz, cx0:cx0+crop_sz]
    crop_resid = residual[cy0:cy0+crop_sz, cx0:cx0+crop_sz]
    crop_bin   = binary[cy0:cy0+crop_sz, cx0:cx0+crop_sz]

    # annotated crop
    crop_ann = cv2.cvtColor(crop_scene, cv2.COLOR_GRAY2BGR)
    for d in dets:
        dx = d.get("cx_refined", d["cx"]) - cx0
        dy = d.get("cy_refined", d["cy"]) - cy0
        if 5 < dx < crop_sz - 5 and 5 < dy < crop_sz - 5:
            cv2.circle(crop_ann, (int(dx), int(dy)), 10, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.drawMarker(crop_ann, (int(dx), int(dy)), (0, 0, 255),
                           cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)

    # scale each crop up 2x so details are visible
    s = 2
    c1 = cv2.resize(to_bgr(crop_scene), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    c2 = cv2.resize(to_bgr(crop_resid), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    c3 = cv2.resize(to_bgr(crop_bin),   None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    c4 = cv2.resize(crop_ann,            None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)

    c1 = label_image(c1, "Raw (zoomed)", scale=0.5)
    c2 = label_image(c2, "Residual", scale=0.5)
    c3 = label_image(c3, "Binary mask", scale=0.5)
    c4 = label_image(c4, "Centroids", scale=0.5)

    star_detail = np.hstack([c1, c2, c3, c4])
    detail_title = np.zeros((30, star_detail.shape[1], 3), dtype=np.uint8)
    cv2.putText(detail_title, "Star Detection Detail (200x200 crop, 2x zoom)  |  Green: detection ring  |  Red cross: refined centroid",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    star_detail = np.vstack([detail_title, star_detail])

    detail_path = os.path.join(out_dir, 'star_detection_detail.png')
    cv2.imwrite(detail_path, star_detail)
    print(f"  -> {detail_path}")

    # =========================================================
    #  PANEL 3: Streak detection detail
    # =========================================================
    streak_vis = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # raw hough in yellow on edge map
    for seg in raw_segs:
        cv2.line(edge_vis, (seg["x1"], seg["y1"]),
                 (seg["x2"], seg["y2"]), (0, 220, 255), 1, cv2.LINE_AA)

    # merged on original
    for s in merged_streaks:
        cv2.line(streak_vis, (s["x1"], s["y1"]),
                 (s["x2"], s["y2"]), (80, 80, 255), 2, cv2.LINE_AA)
        # label length
        mid_x = (s["x1"] + s["x2"]) // 2
        mid_y = (s["y1"] + s["y2"]) // 2
        cv2.putText(streak_vis, f"{s['length']:.0f}px",
                    (mid_x + 5, mid_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (80, 80, 255), 1, cv2.LINE_AA)

    # ground truth overlay (dashed isn't easy in opencv, use dotted)
    for (x1, y1, x2, y2, _) in streak_gt:
        for t in np.linspace(0, 1, 40):
            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))
            if int(t * 40) % 3 == 0:
                cv2.circle(streak_vis, (px, py), 1, (0, 180, 0), -1)

    e_labeled = label_image(edge_vis,   "Edges + Hough segments (yellow)")
    s_labeled = label_image(streak_vis, "Merged streaks (red) vs ground truth (green dots)")

    streak_panel = np.hstack([e_labeled, s_labeled])
    streak_title = np.zeros((30, streak_panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(streak_title, "Streak Detection Results",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    streak_panel = np.vstack([streak_title, streak_panel])

    streak_path = os.path.join(out_dir, 'streak_detection_detail.png')
    cv2.imwrite(streak_path, streak_panel)
    print(f"  -> {streak_path}")

    # =========================================================
    #  PANEL 4: single combined "hero" output image
    # =========================================================
    # full composite with everything annotated + stats text
    hero = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)

    # stars
    for d in dets:
        cx = d.get("cx_refined", d["cx"])
        cy = d.get("cy_refined", d["cy"])
        cv2.circle(hero, (int(cx), int(cy)), 7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(hero, (int(cx), int(cy)), 1, (0, 180, 255), -1, cv2.LINE_AA)

    # streaks
    for s in merged_streaks:
        cv2.line(hero, (s["x1"], s["y1"]),
                 (s["x2"], s["y2"]), (80, 80, 255), 2, cv2.LINE_AA)

    # stats box in top-right
    stats_lines = [
        f"Stars detected: {len(dets)} / {len(star_gt)} planted",
        f"Streaks detected: {len(merged_streaks)} / {len(streak_gt)} planted",
        f"Image: 512x512 synthetic",
    ]
    box_x, box_y = 270, 8
    cv2.rectangle(hero, (box_x - 5, box_y - 2),
                  (510, box_y + 15 * len(stats_lines) + 5), (0, 0, 0), -1)
    for i, line in enumerate(stats_lines):
        cv2.putText(hero, line, (box_x, box_y + 13 + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 200), 1, cv2.LINE_AA)

    # legend in bottom-left
    legend_y = 470
    cv2.rectangle(hero, (5, legend_y - 2), (220, 510), (0, 0, 0), -1)
    cv2.circle(hero, (15, legend_y + 10), 5, (0, 255, 0), 1)
    cv2.putText(hero, "Star detection", (25, legend_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.line(hero, (10, legend_y + 28), (30, legend_y + 28), (80, 80, 255), 2)
    cv2.putText(hero, "Streak detection", (35, legend_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 255), 1)

    hero_path = os.path.join(out_dir, 'detection_output.png')
    cv2.imwrite(hero_path, hero)
    print(f"  -> {hero_path}")

    print("\nAll visualizations saved to data/")
    print("  pipeline_stages.png        - 2x3 grid of every pipeline stage")
    print("  star_detection_detail.png  - zoomed crop of star centroid results")
    print("  streak_detection_detail.png - edge map + merged streak overlay")
    print("  detection_output.png       - final composite output with stats")


if __name__ == "__main__":
    main()
