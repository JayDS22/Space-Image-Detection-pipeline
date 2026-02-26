/*
 * streak_detector.hpp
 *
 * Satellite trail / debris streak detection via Hough + clustering.
 * Ported from python_prototypes/streak_detector.py.
 *
 * The clustering is O(n^2) on the number of Hough segments, which is
 * totally fine for typical frames (< ~200 segments). If that ever
 * becomes a problem we can swap in a kd-tree or DBSCAN approach.
 */

#ifndef STREAK_DETECTOR_HPP
#define STREAK_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <vector>

namespace sslab {

struct LineSegment {
    int    x1, y1, x2, y2;
    double length;
    double angle_rad;
};

struct StreakDetection {
    int    x1, y1, x2, y2;
    double length;
    int    num_fragments;   // how many raw Hough pieces got merged
};

struct StreakTimingInfo {
    double preprocess_ms;
    double edge_detect_ms;
    double hough_ms;
    double clustering_ms;
    double total_ms;
};

struct StreakDetectorConfig {
    int    morph_kernel_size  = 21;
    int    canny_low          = 30;
    int    canny_high         = 90;
    int    hough_threshold    = 40;
    int    hough_min_length   = 30;
    int    hough_max_gap      = 10;
    double cluster_angle_tol  = 5.0;   // degrees
    double cluster_dist_tol   = 15.0;  // px
};

// End-to-end streak detection on a single grayscale frame.
// Optionally returns the pre-clustering Hough segments and timing data.
std::vector<StreakDetection> detect_streaks(
    const cv::Mat& image,
    const StreakDetectorConfig& config,
    std::vector<LineSegment>* raw_segments_out = nullptr,
    StreakTimingInfo* timing = nullptr);

// Groups nearly-collinear segments into single streaks based on
// angle similarity and perpendicular distance between midpoints.
std::vector<StreakDetection> cluster_segments(
    const std::vector<LineSegment>& segments,
    double angle_tol_deg = 5.0,
    double dist_tol = 15.0);

// Draws a few random linear streaks onto an image for testing.
cv::Mat add_synthetic_streaks(
    const cv::Mat& image,
    int num_streaks,
    std::vector<std::array<int, 4>>& streak_endpoints,
    unsigned int seed = 123);

} // namespace sslab

#endif // STREAK_DETECTOR_HPP
