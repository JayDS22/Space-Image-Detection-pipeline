/*
 * star_detector.hpp
 *
 * Point-source detection + sub-pixel centroid extraction.
 * This is the C++ port of python_prototypes/star_detector.py; the main
 * win here over the Python version is avoiding per-detection heap allocs
 * during centroid refinement (we just index into the residual directly
 * instead of slicing out numpy arrays).
 *
 * Each function is safe to call from multiple threads as long as they
 * aren't sharing the same cv::Mat buffer.
 */

#ifndef STAR_DETECTOR_HPP
#define STAR_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace sslab {

// One detected point source. The "raw" centroid is whatever
// connectedComponentsWithStats gives us; "refined" is after
// intensity-weighted moment correction (usually gets us to
// ~0.1-0.3 px accuracy on decently-sampled PSFs).
struct StarDetection {
    int    label;
    double cx_raw;
    double cy_raw;
    double cx_refined;
    double cy_refined;
    int    area;           // blob area (px)
    int    bbox_w;
    int    bbox_h;
};

// Timing breakdown so we can see where the bottlenecks are.
struct StarTimingInfo {
    double background_ms;
    double threshold_ms;
    double component_filter_ms;
    double centroid_refine_ms;
    double total_ms;
};

// Defaults are set to match the Python prototype so we can
// cross-validate easily. Tweak as needed for real sensor data.
struct StarDetectorConfig {
    int    morph_kernel_size  = 21;
    double sigma_multiplier   = 3.5;
    int    min_area           = 3;
    int    max_area           = 500;
    double max_eccentricity   = 0.85;
    int    centroid_half_win   = 4;
};

// Full pipeline: bg estimation -> threshold -> cc labeling -> centroid refine.
// Pass a StarTimingInfo* if you want per-stage timing numbers.
std::vector<StarDetection> detect_stars(
    const cv::Mat& image,
    const StarDetectorConfig& config,
    StarTimingInfo* timing = nullptr);

// Runs connected components and throws out blobs that are too big,
// too small, or too elongated to be real stars.
std::vector<StarDetection> filter_components(
    const cv::Mat& binary,
    int min_area,
    int max_area,
    double max_eccentricity);

// Nudges each centroid to sub-pixel precision using intensity-weighted
// moments in a small window around the initial position.
void refine_centroids(
    const cv::Mat& residual,
    std::vector<StarDetection>& detections,
    int half_window = 4);

// Makes a fake star field for testing. Populates ground_truth with
// {x, y, brightness, sigma} for each planted star so we can compute recall.
cv::Mat generate_synthetic_starfield(
    int width, int height, int num_stars,
    double noise_std,
    std::vector<std::array<double, 4>>& ground_truth,
    unsigned int seed = 42);

} // namespace sslab

#endif // STAR_DETECTOR_HPP
