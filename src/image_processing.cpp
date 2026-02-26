/*
 * image_processing.cpp
 *
 * Background estimation + adaptive thresholding -- the two steps that
 * both detectors share. These eat a good chunk of the per-frame budget
 * (~40-60% on a 1024x1024 frame), mostly because the morph opening
 * has to iterate a big structuring element across the whole image.
 */

#include "image_processing.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace sslab {

cv::Mat estimate_background(const cv::Mat& image, int kernel_size) {
    // Elliptical SE works better than a rectangle for round sources --
    // you get fewer artifacts at the boundary of the opening.
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));

    cv::Mat background;
    cv::morphologyEx(image, background, cv::MORPH_OPEN, kernel);
    return background;
}


cv::Mat adaptive_threshold(const cv::Mat& image,
                           const cv::Mat& background,
                           double sigma_multiplier,
                           cv::Mat* residual_out) {
    // cv::subtract clamps to 0 for uint8, no wraparound worries
    cv::Mat residual;
    cv::subtract(image, background, residual);

    if (residual_out != nullptr) {
        residual.copyTo(*residual_out);
    }

    // We want noise stats from the "quiet" part of the image only.
    // Bright star pixels would skew the mean/sigma upward, so chop
    // them off at the 90th percentile before computing stats.
    std::vector<uchar> pixels;
    pixels.reserve(residual.total());

    // flatten into a vector we can partial-sort
    if (residual.isContinuous()) {
        pixels.assign(residual.data, residual.data + residual.total());
    } else {
        for (int r = 0; r < residual.rows; ++r) {
            const uchar* row = residual.ptr<uchar>(r);
            pixels.insert(pixels.end(), row, row + residual.cols);
        }
    }

    // 90th percentile via partial sort
    size_t cutoff_idx = static_cast<size_t>(pixels.size() * 0.90);
    std::nth_element(pixels.begin(), pixels.begin() + cutoff_idx, pixels.end());
    uchar cutoff_val = pixels[cutoff_idx];

    // mean + stddev of everything below cutoff
    double sum = 0.0;
    double sq_sum = 0.0;
    int count = 0;
    for (uchar val : pixels) {
        if (val <= cutoff_val) {
            double v = static_cast<double>(val);
            sum += v;
            sq_sum += v * v;
            ++count;
        }
    }

    double mu = (count > 0) ? sum / count : 0.0;
    double variance = (count > 1)
                      ? (sq_sum / count - mu * mu)
                      : 0.0;
    double sigma = std::sqrt(std::max(variance, 0.0));

    double thresh = mu + sigma_multiplier * sigma;
    thresh = std::max(thresh, 1.0);  // floor -- zero threshold is useless

    cv::Mat binary;
    cv::threshold(residual, binary, thresh, 255, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8UC1);

    return binary;
}

} // namespace sslab
