/*
 * image_processing.hpp
 *
 * Shared low-level routines used by both the star and streak detectors.
 * Ported from the Python prototypes, with an eye toward eventually
 * moving the hot paths (morph opening, bilateral) to CUDA.
 *
 * Coords: (x,y), origin top-left, standard OpenCV layout.
 */

#ifndef IMAGE_PROCESSING_HPP
#define IMAGE_PROCESSING_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>

namespace sslab {

// Morphological opening to get a smooth background estimate.
// kernel_size needs to be bigger than the widest PSF you expect,
// otherwise the opening won't fully erode the stars away.
cv::Mat estimate_background(const cv::Mat& image, int kernel_size = 21);

// Subtracts background, then picks a threshold at (mean + k*sigma) of
// the noise floor. Ignores the brightest 10% of pixels when computing
// stats so dense fields don't bias the threshold upward.
// Optionally hands back the residual through residual_out.
cv::Mat adaptive_threshold(const cv::Mat& image,
                           const cv::Mat& background,
                           double sigma_multiplier = 3.5,
                           cv::Mat* residual_out = nullptr);

} // namespace sslab

#endif // IMAGE_PROCESSING_HPP
