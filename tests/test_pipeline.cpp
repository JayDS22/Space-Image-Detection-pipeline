/*
 * test_pipeline.cpp
 *
 * Smoke tests + a couple integration tests for the detection pipeline.
 * No external framework -- just bool-returning functions and a macro
 * to track pass/fail counts. Run with: ./test_pipeline
 */

#include "star_detector.hpp"
#include "streak_detector.hpp"
#include "image_processing.hpp"
#include "profiler.hpp"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <cassert>

using namespace sslab;

static int tests_passed = 0;
static int tests_failed = 0;

#define RUN_TEST(name) do { \
    std::cout << "  " << #name << "... "; \
    if (name()) { \
        std::cout << "PASSED\n"; \
        ++tests_passed; \
    } else { \
        std::cout << "FAILED\n"; \
        ++tests_failed; \
    } \
} while(0)


/* ---- Background estimation tests ---- */

bool test_background_removes_small_features() {
    // single bright pixel on black -- the opening should wipe it out
    cv::Mat img = cv::Mat::zeros(64, 64, CV_8UC1);
    img.at<uchar>(32, 32) = 200;

    cv::Mat bg = estimate_background(img, 11);
    return bg.at<uchar>(32, 32) < 10;
}

bool test_background_preserves_gradients() {
    // a smooth ramp should mostly survive the opening
    cv::Mat img(128, 128, CV_8UC1);
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x)
            img.at<uchar>(y, x) = static_cast<uchar>(x * 2);

    cv::Mat bg = estimate_background(img, 15);

    // allow some error near boundaries
    double diff = std::abs(static_cast<double>(bg.at<uchar>(64, 64))
                          - static_cast<double>(img.at<uchar>(64, 64)));
    return diff < 20;
}


/* ---- Adaptive threshold tests ---- */

bool test_threshold_detects_bright_spots() {
    // bright blob on dim background should produce nonzero binary pixels
    cv::Mat img = cv::Mat::ones(128, 128, CV_8UC1) * 30;
    cv::circle(img, cv::Point(64, 64), 3, cv::Scalar(180), -1);

    cv::Mat bg = estimate_background(img, 21);
    cv::Mat binary = adaptive_threshold(img, bg, 3.0);

    cv::Rect roi(60, 60, 9, 9);
    double region_sum = cv::sum(binary(roi))[0];
    return region_sum > 0;
}

bool test_threshold_rejects_noise() {
    // flat image with tight noise band -> threshold should give almost nothing
    cv::Mat img(128, 128, CV_8UC1);
    cv::randu(img, 25, 35);

    cv::Mat bg = estimate_background(img, 21);
    cv::Mat binary = adaptive_threshold(img, bg, 5.0);  // strict

    double total = cv::sum(binary)[0];
    return total < 255.0 * 20;  // fewer than 20 false positives
}


/* ---- Star detection integration test ---- */

bool test_star_detection_recall() {
    // plant 40 stars, run the full pipeline, check we find most of them
    std::vector<std::array<double, 4>> ground_truth;
    cv::Mat image = generate_synthetic_starfield(512, 512, 40, 10.0,
                                                  ground_truth, 77);

    StarDetectorConfig config;
    auto detections = detect_stars(image, config);

    // match each GT star to nearest detection
    int matched = 0;
    for (const auto& gt : ground_truth) {
        for (const auto& det : detections) {
            double dist = std::hypot(det.cx_refined - gt[0],
                                      det.cy_refined - gt[1]);
            if (dist < 4.0) {
                ++matched;
                break;
            }
        }
    }

    double recall = static_cast<double>(matched) / ground_truth.size();
    // 60% is a conservative bar; the dimmer stars near the noise floor
    // won't always make it through, which is expected
    return recall > 0.60;
}

bool test_star_centroid_accuracy() {
    // one bright star at a known sub-pixel position -- verify the
    // refined centroid lands within 1 px
    cv::Mat img = cv::Mat::ones(128, 128, CV_8UC1) * 30;

    double true_x = 64.3, true_y = 64.7;
    for (int y = 56; y < 73; ++y) {
        for (int x = 56; x < 73; ++x) {
            double dx = x - true_x;
            double dy = y - true_y;
            double val = 180.0 * std::exp(-(dx*dx + dy*dy) / (2.0 * 2.0 * 2.0));
            img.at<uchar>(y, x) = static_cast<uchar>(
                std::min(255.0, 30.0 + val));
        }
    }

    StarDetectorConfig config;
    auto detections = detect_stars(img, config);

    if (detections.empty()) return false;

    // find closest detection to our planted position
    double best_dist = 1e6;
    for (const auto& det : detections) {
        double dist = std::hypot(det.cx_refined - true_x,
                                  det.cy_refined - true_y);
        best_dist = std::min(best_dist, dist);
    }

    return best_dist < 1.0;
}


/* ---- Streak detection tests ---- */

bool test_streak_detection_finds_lines() {
    // obvious diagonal line on dark background -- should be trivial
    cv::Mat img = cv::Mat::ones(256, 256, CV_8UC1) * 20;
    cv::line(img, cv::Point(30, 30), cv::Point(220, 200),
             cv::Scalar(180), 2, cv::LINE_AA);

    StreakDetectorConfig config;
    config.hough_threshold = 30;
    config.hough_min_length = 20;

    auto streaks = detect_streaks(img, config);
    return !streaks.empty();
}

bool test_cluster_merges_colinear() {
    // two segments along the same 45-degree line -- should fuse into one
    std::vector<LineSegment> segments = {
        {10, 10, 50, 50, 0, M_PI / 4.0},
        {55, 55, 90, 90, 0, M_PI / 4.0},
    };
    for (auto& s : segments) {
        s.length = std::hypot(s.x2 - s.x1, s.y2 - s.y1);
    }

    auto merged = cluster_segments(segments, 10.0, 20.0);
    return merged.size() == 1 && merged[0].num_fragments == 2;
}

bool test_cluster_separates_perpendicular() {
    // horizontal + vertical should stay as two separate detections
    std::vector<LineSegment> segments = {
        {10, 50, 90, 50, 80.0, 0.0},
        {50, 10, 50, 90, 80.0, M_PI / 2.0},
    };

    auto merged = cluster_segments(segments, 5.0, 15.0);
    return merged.size() == 2;
}


/* ---- Profiler ---- */

bool test_scoped_timer() {
    double elapsed = 0;
    {
        ScopedTimer timer(elapsed);
        volatile int x = 0;
        for (int i = 0; i < 100000; ++i) x += i;
    }
    return elapsed > 0.0;
}

bool test_benchmark_stats() {
    PipelineBenchmark bench;
    bench.record("test", 10.0);
    bench.record("test", 20.0);
    bench.record("test", 30.0);

    auto stats = bench.get_stats("test");
    bool mean_ok = std::abs(stats.mean_ms - 20.0) < 0.01;
    bool min_ok = std::abs(stats.min_ms - 10.0) < 0.01;
    bool max_ok = std::abs(stats.max_ms - 30.0) < 0.01;
    bool count_ok = stats.num_runs == 3;

    return mean_ok && min_ok && max_ok && count_ok;
}


int main() {
    std::cout << "Running pipeline tests...\n\n";

    std::cout << "[Background Estimation]\n";
    RUN_TEST(test_background_removes_small_features);
    RUN_TEST(test_background_preserves_gradients);

    std::cout << "\n[Adaptive Threshold]\n";
    RUN_TEST(test_threshold_detects_bright_spots);
    RUN_TEST(test_threshold_rejects_noise);

    std::cout << "\n[Star Detection]\n";
    RUN_TEST(test_star_detection_recall);
    RUN_TEST(test_star_centroid_accuracy);

    std::cout << "\n[Streak Detection]\n";
    RUN_TEST(test_streak_detection_finds_lines);
    RUN_TEST(test_cluster_merges_colinear);
    RUN_TEST(test_cluster_separates_perpendicular);

    std::cout << "\n[Profiler]\n";
    RUN_TEST(test_scoped_timer);
    RUN_TEST(test_benchmark_stats);

    std::cout << "\n========================================\n"
              << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed\n"
              << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}
