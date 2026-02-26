/*
 * main.cpp
 *
 * CLI front-end for the detection pipeline. Three ways to run it:
 *   ./space_sensing_pipeline                        (synthetic test data)
 *   ./space_sensing_pipeline --input frame.png      (real image)
 *   ./space_sensing_pipeline --benchmark --runs 50  (profiling)
 */

#include "star_detector.hpp"
#include "streak_detector.hpp"
#include "profiler.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <cmath>

using namespace sslab;

struct ProgramArgs {
    std::string input_path;
    std::string output_stars_image;
    std::string output_streaks_image;
    bool synthetic     = true;
    bool benchmark     = false;
    int  benchmark_runs = 20;
    bool help          = false;
};

ProgramArgs parse_args(int argc, char** argv) {
    ProgramArgs args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            args.input_path = argv[++i];
            args.synthetic = false;
        } else if (std::strcmp(argv[i], "--output-stars") == 0 && i + 1 < argc) {
            args.output_stars_image = argv[++i];
        } else if (std::strcmp(argv[i], "--output-streaks") == 0 && i + 1 < argc) {
            args.output_streaks_image = argv[++i];
        } else if (std::strcmp(argv[i], "--benchmark") == 0) {
            args.benchmark = true;
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            args.benchmark_runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            args.help = true;
        }
    }
    return args;
}

void print_usage() {
    std::cout << "Space Sensing Image Processing Pipeline\n"
              << "Usage:\n"
              << "  space_sensing_pipeline [options]\n\n"
              << "Options:\n"
              << "  --input <path>          Input grayscale image (PNG/TIFF)\n"
              << "  --output-stars <path>   Save annotated star detection image\n"
              << "  --output-streaks <path> Save annotated streak detection image\n"
              << "  --benchmark             Run profiling mode\n"
              << "  --runs <N>              Number of benchmark iterations (default: 20)\n"
              << "  -h, --help              Show this message\n";
}


void run_star_detection(const cv::Mat& image,
                        const std::vector<std::array<double, 4>>* ground_truth,
                        const std::string& output_path) {
    StarDetectorConfig config;
    StarTimingInfo timing;

    auto detections = detect_stars(image, config, &timing);

    std::cout << "\n=== Star Detection Results ===\n"
              << "Detections found: " << detections.size() << "\n"
              << std::fixed << std::setprecision(2)
              << "  Background estimation: " << timing.background_ms << " ms\n"
              << "  Adaptive threshold:    " << timing.threshold_ms << " ms\n"
              << "  Component filtering:   " << timing.component_filter_ms << " ms\n"
              << "  Centroid refinement:   " << timing.centroid_refine_ms << " ms\n"
              << "  Total:                 " << timing.total_ms << " ms\n";

    // compute recall against planted positions if we have ground truth
    if (ground_truth != nullptr && !ground_truth->empty()) {
        int matched = 0;
        for (const auto& gt : *ground_truth) {
            double gx = gt[0], gy = gt[1];
            for (const auto& det : detections) {
                double dist = std::hypot(det.cx_refined - gx,
                                          det.cy_refined - gy);
                if (dist < 3.0) {
                    ++matched;
                    break;
                }
            }
        }
        double recall = static_cast<double>(matched)
                        / ground_truth->size() * 100.0;
        std::cout << "  Recall: " << matched << "/"
                  << ground_truth->size()
                  << " (" << std::setprecision(1) << recall << "%)\n";
    }

    // draw green circles on detected stars and save
    if (!output_path.empty()) {
        cv::Mat vis;
        cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
        for (const auto& det : detections) {
            cv::Point center(static_cast<int>(det.cx_refined),
                             static_cast<int>(det.cy_refined));
            cv::circle(vis, center, 8, cv::Scalar(0, 255, 0), 1);
            cv::circle(vis, center, 1, cv::Scalar(0, 0, 255), -1);
        }
        cv::imwrite(output_path, vis);
        std::cout << "  Annotated image saved to: " << output_path << "\n";
    }
}


void run_streak_detection(const cv::Mat& image,
                          const std::string& output_path) {
    StreakDetectorConfig config;
    StreakTimingInfo timing;
    std::vector<LineSegment> raw_segments;

    auto streaks = detect_streaks(image, config, &raw_segments, &timing);

    std::cout << "\n=== Streak Detection Results ===\n"
              << "Raw Hough segments: " << raw_segments.size() << "\n"
              << "Merged streaks:     " << streaks.size() << "\n"
              << std::fixed << std::setprecision(2)
              << "  Preprocessing:   " << timing.preprocess_ms << " ms\n"
              << "  Edge detection:  " << timing.edge_detect_ms << " ms\n"
              << "  Hough transform: " << timing.hough_ms << " ms\n"
              << "  Clustering:      " << timing.clustering_ms << " ms\n"
              << "  Total:           " << timing.total_ms << " ms\n";

    for (size_t i = 0; i < streaks.size(); ++i) {
        const auto& s = streaks[i];
        std::cout << "  Streak " << i << ": ("
                  << s.x1 << "," << s.y1 << ")->(" << s.x2 << "," << s.y2
                  << ")  len=" << std::setprecision(1) << s.length
                  << "px  fragments=" << s.num_fragments << "\n";
    }

    if (!output_path.empty()) {
        cv::Mat vis;
        cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
        // raw hough segments in dim blue, merged in bright green
        for (const auto& seg : raw_segments) {
            cv::line(vis, cv::Point(seg.x1, seg.y1),
                     cv::Point(seg.x2, seg.y2),
                     cv::Scalar(180, 100, 50), 1);
        }
        for (const auto& s : streaks) {
            cv::line(vis, cv::Point(s.x1, s.y1), cv::Point(s.x2, s.y2),
                     cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite(output_path, vis);
        std::cout << "  Annotated image saved to: " << output_path << "\n";
    }
}


void run_benchmark(const cv::Mat& image, int num_runs) {
    std::cout << "\n=== Benchmark Mode (" << num_runs << " runs) ===\n";
    PipelineBenchmark bench;

    StarDetectorConfig star_cfg;
    StreakDetectorConfig streak_cfg;

    for (int r = 0; r < num_runs; ++r) {
        StarTimingInfo st;
        detect_stars(image, star_cfg, &st);
        bench.record("star_background", st.background_ms);
        bench.record("star_threshold", st.threshold_ms);
        bench.record("star_components", st.component_filter_ms);
        bench.record("star_centroids", st.centroid_refine_ms);
        bench.record("star_total", st.total_ms);

        StreakTimingInfo skt;
        detect_streaks(image, streak_cfg, nullptr, &skt);
        bench.record("streak_preprocess", skt.preprocess_ms);
        bench.record("streak_edges", skt.edge_detect_ms);
        bench.record("streak_hough", skt.hough_ms);
        bench.record("streak_cluster", skt.clustering_ms);
        bench.record("streak_total", skt.total_ms);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(22) << "Stage"
              << std::right
              << std::setw(10) << "Mean(ms)"
              << std::setw(10) << "Std(ms)"
              << std::setw(10) << "Min(ms)"
              << std::setw(10) << "Max(ms)" << "\n";
    std::cout << std::string(62, '-') << "\n";

    for (const auto& name : bench.stage_names()) {
        auto s = bench.get_stats(name);
        std::cout << std::left << std::setw(22) << name
                  << std::right
                  << std::setw(10) << s.mean_ms
                  << std::setw(10) << s.stddev_ms
                  << std::setw(10) << s.min_ms
                  << std::setw(10) << s.max_ms << "\n";
    }
}


int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    if (args.help) {
        print_usage();
        return 0;
    }

    cv::Mat image;
    std::vector<std::array<double, 4>> star_gt;
    std::vector<std::array<int, 4>> streak_gt;

    if (args.synthetic) {
        std::cout << "[INFO] Generating synthetic 1024x1024 test scene...\n";
        image = generate_synthetic_starfield(1024, 1024, 80, 12.0, star_gt, 42);
        image = add_synthetic_streaks(image, 4, streak_gt, 123);
        std::cout << "[INFO] Planted " << star_gt.size() << " stars and "
                  << streak_gt.size() << " streaks\n";
    } else {
        image = cv::imread(args.input_path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "[ERROR] Could not load image: "
                      << args.input_path << "\n";
            return 1;
        }
        std::cout << "[INFO] Loaded " << image.cols << "x" << image.rows
                  << " image from " << args.input_path << "\n";
    }

    if (args.benchmark) {
        run_benchmark(image, args.benchmark_runs);
    } else {
        const auto* gt_ptr = args.synthetic ? &star_gt : nullptr;
        run_star_detection(image, gt_ptr, args.output_stars_image);
        run_streak_detection(image, args.output_streaks_image);
    }

    return 0;
}
