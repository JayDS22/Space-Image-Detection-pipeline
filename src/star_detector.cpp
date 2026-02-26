/*
 * star_detector.cpp
 *
 * C++ port of the Python star detector. The algorithm is identical but
 * we avoid a lot of overhead from the numpy side:
 *   - centroid ROIs are just pointer arithmetic, no temp array copies
 *   - CC stats come straight from OpenCV's output mat, no dict building
 *   - synthetic starfield RNG uses <random> (same seed -> same scene,
 *     verified against the Python output)
 */

#include "star_detector.hpp"
#include "image_processing.hpp"
#include "profiler.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <random>
#include <algorithm>

namespace sslab {

std::vector<StarDetection> filter_components(
    const cv::Mat& binary,
    int min_area,
    int max_area,
    double max_eccentricity) {

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    std::vector<StarDetection> detections;
    detections.reserve(num_labels);  // won't need all of these

    // skip label 0 (that's the background)
    for (int i = 1; i < num_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < min_area || area > max_area) {
            continue;
        }

        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // rough eccentricity from bounding box: circles have aspect ~1,
        // cosmic ray streaks have it near 0
        double max_dim = static_cast<double>(std::max(w, h));
        double min_dim = static_cast<double>(std::min(w, h));
        double aspect = (max_dim > 0) ? (min_dim / max_dim) : 1.0;

        if ((1.0 - aspect) > max_eccentricity) {
            continue;
        }

        StarDetection det;
        det.label = i;
        det.cx_raw = centroids.at<double>(i, 0);
        det.cy_raw = centroids.at<double>(i, 1);
        det.cx_refined = det.cx_raw;   // will be updated later
        det.cy_refined = det.cy_raw;
        det.area = area;
        det.bbox_w = w;
        det.bbox_h = h;
        detections.push_back(det);
    }

    return detections;
}


void refine_centroids(
    const cv::Mat& residual,
    std::vector<StarDetection>& detections,
    int half_window) {

    int rows = residual.rows;
    int cols = residual.cols;

    for (auto& det : detections) {
        int cx_int = static_cast<int>(det.cx_raw);
        int cy_int = static_cast<int>(det.cy_raw);

        // keep the window inside the image
        int x0 = std::max(cx_int - half_window, 0);
        int x1 = std::min(cx_int + half_window + 1, cols);
        int y0 = std::max(cy_int - half_window, 0);
        int y1 = std::min(cy_int + half_window + 1, rows);

        // first moment of intensity -> weighted centroid
        double sum_w = 0.0;
        double sum_wx = 0.0;
        double sum_wy = 0.0;

        for (int y = y0; y < y1; ++y) {
            const uchar* row_ptr = residual.ptr<uchar>(y);
            for (int x = x0; x < x1; ++x) {
                double w = static_cast<double>(row_ptr[x]);
                sum_w  += w;
                sum_wx += w * x;
                sum_wy += w * y;
            }
        }

        if (sum_w > 1e-6) {
            det.cx_refined = sum_wx / sum_w;
            det.cy_refined = sum_wy / sum_w;
        }
        // if there's basically no signal, just keep the raw position
    }
}


std::vector<StarDetection> detect_stars(
    const cv::Mat& image,
    const StarDetectorConfig& config,
    StarTimingInfo* timing) {

    double t_bg = 0, t_thresh = 0, t_comp = 0, t_refine = 0;

    cv::Mat background, binary, residual;

    {
        ScopedTimer timer(t_bg);
        background = estimate_background(image, config.morph_kernel_size);
    }

    {
        ScopedTimer timer(t_thresh);
        binary = adaptive_threshold(image, background,
                                    config.sigma_multiplier, &residual);
    }

    std::vector<StarDetection> detections;
    {
        ScopedTimer timer(t_comp);
        detections = filter_components(binary, config.min_area,
                                       config.max_area,
                                       config.max_eccentricity);
    }

    {
        ScopedTimer timer(t_refine);
        refine_centroids(residual, detections, config.centroid_half_win);
    }

    if (timing != nullptr) {
        timing->background_ms = t_bg;
        timing->threshold_ms = t_thresh;
        timing->component_filter_ms = t_comp;
        timing->centroid_refine_ms = t_refine;
        timing->total_ms = t_bg + t_thresh + t_comp + t_refine;
    }

    return detections;
}


cv::Mat generate_synthetic_starfield(
    int width, int height, int num_stars,
    double noise_std,
    std::vector<std::array<double, 4>>& ground_truth,
    unsigned int seed) {

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos_x(20.0, width - 20.0);
    std::uniform_real_distribution<double> pos_y(20.0, height - 20.0);
    std::uniform_real_distribution<double> brightness(60.0, 220.0);
    std::uniform_real_distribution<double> spread(1.2, 2.5);
    std::normal_distribution<double> noise(0.0, noise_std);

    // smooth sinusoidal gradient to mimic uneven sensor illumination
    cv::Mat image(height, width, CV_64FC1);
    for (int y = 0; y < height; ++y) {
        double* row = image.ptr<double>(y);
        for (int x = 0; x < width; ++x) {
            row[x] = 30.0
                     + 20.0 * std::sin(2.0 * M_PI * x / width)
                     + 15.0 * std::cos(2.0 * M_PI * y / height);
        }
    }

    // drop in Gaussian PSFs at random positions
    ground_truth.clear();
    ground_truth.reserve(num_stars);

    for (int s = 0; s < num_stars; ++s) {
        double sx = pos_x(rng);
        double sy = pos_y(rng);
        double b  = brightness(rng);
        double sig = spread(rng);

        ground_truth.push_back({sx, sy, b, sig});

        // stamp a small Gaussian around the star center
        int stamp_half = 8;
        int y_lo = std::max(0, static_cast<int>(sy) - stamp_half);
        int y_hi = std::min(height, static_cast<int>(sy) + stamp_half + 1);
        int x_lo = std::max(0, static_cast<int>(sx) - stamp_half);
        int x_hi = std::min(width, static_cast<int>(sx) + stamp_half + 1);

        for (int y = y_lo; y < y_hi; ++y) {
            double* row = image.ptr<double>(y);
            for (int x = x_lo; x < x_hi; ++x) {
                double dx = x - sx;
                double dy = y - sy;
                double gauss = b * std::exp(-(dx*dx + dy*dy) / (2.0 * sig * sig));
                row[x] += gauss;
            }
        }
    }

    // add read noise
    for (int y = 0; y < height; ++y) {
        double* row = image.ptr<double>(y);
        for (int x = 0; x < width; ++x) {
            row[x] += noise(rng);
            row[x] = std::clamp(row[x], 0.0, 255.0);
        }
    }

    // Convert to 8-bit
    cv::Mat result;
    image.convertTo(result, CV_8UC1);
    return result;
}

} // namespace sslab
