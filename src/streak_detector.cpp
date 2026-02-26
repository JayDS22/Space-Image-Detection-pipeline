/*
 * streak_detector.cpp
 *
 * C++ port of the streak/satellite trail detector. The trickiest bit
 * was the clustering -- Python gets to lean on numpy broadcasting for
 * the angle/distance checks, but here it's just nested loops. Doesn't
 * matter performance-wise though; profiling shows the clustering step
 * is <0.5ms even with 150+ segments. The real cost is in the bilateral
 * filter and Hough transform.
 */

#include "streak_detector.hpp"
#include "image_processing.hpp"
#include "profiler.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace sslab {

namespace {

// Normalize angle difference to [0, pi/2]. Two lines pointing in
// opposite directions are still parallel.
double angle_diff(double a, double b) {
    double d = std::abs(a - b);
    d = std::fmod(d, M_PI);
    if (d > M_PI / 2.0) d = M_PI - d;
    return d;
}

} // anonymous namespace


std::vector<StreakDetection> cluster_segments(
    const std::vector<LineSegment>& segments,
    double angle_tol_deg,
    double dist_tol) {

    if (segments.empty()) return {};

    double angle_tol = angle_tol_deg * M_PI / 180.0;
    size_t n = segments.size();
    std::vector<bool> visited(n, false);
    std::vector<StreakDetection> merged;

    for (size_t i = 0; i < n; ++i) {
        if (visited[i]) continue;
        visited[i] = true;

        std::vector<size_t> cluster_indices = {i};

        double mid_ix = (segments[i].x1 + segments[i].x2) / 2.0;
        double mid_iy = (segments[i].y1 + segments[i].y2) / 2.0;
        double ang_i = segments[i].angle_rad;

        for (size_t j = i + 1; j < n; ++j) {
            if (visited[j]) continue;

            // angle check first since it's cheaper
            double d_ang = angle_diff(ang_i, segments[j].angle_rad);
            if (d_ang > angle_tol) continue;

            // how far is midpoint j from the line defined by segment i?
            double mid_jx = (segments[j].x1 + segments[j].x2) / 2.0;
            double mid_jy = (segments[j].y1 + segments[j].y2) / 2.0;
            double dir_x = std::cos(ang_i);
            double dir_y = std::sin(ang_i);

            double dx = mid_jx - mid_ix;
            double dy = mid_jy - mid_iy;
            double perp_dist = std::abs(dx * dir_y - dy * dir_x);

            if (perp_dist <= dist_tol) {
                cluster_indices.push_back(j);
                visited[j] = true;
            }
        }

        // gather all endpoints, then find the major axis via PCA
        std::vector<cv::Point2d> pts;
        pts.reserve(cluster_indices.size() * 2);
        for (size_t idx : cluster_indices) {
            pts.push_back({static_cast<double>(segments[idx].x1),
                           static_cast<double>(segments[idx].y1)});
            pts.push_back({static_cast<double>(segments[idx].x2),
                           static_cast<double>(segments[idx].y2)});
        }

        // Compute mean and covariance for PCA
        cv::Point2d mean(0, 0);
        for (const auto& p : pts) {
            mean.x += p.x;
            mean.y += p.y;
        }
        mean.x /= pts.size();
        mean.y /= pts.size();

        // 2x2 covariance
        double cov_xx = 0, cov_xy = 0, cov_yy = 0;
        for (const auto& p : pts) {
            double dx = p.x - mean.x;
            double dy = p.y - mean.y;
            cov_xx += dx * dx;
            cov_xy += dx * dy;
            cov_yy += dy * dy;
        }

        // analytic eigenvector of the larger eigenvalue for a 2x2 symmetric mat
        double trace = cov_xx + cov_yy;
        double det = cov_xx * cov_yy - cov_xy * cov_xy;
        double disc = std::sqrt(std::max(trace * trace / 4.0 - det, 0.0));
        // double lambda1 = trace / 2.0 + disc;  // larger eigenvalue (unused)

        // major axis eigenvector
        cv::Point2d major_axis;
        if (std::abs(cov_xy) > 1e-8) {
            double lambda1 = trace / 2.0 + disc;
            major_axis = {lambda1 - cov_yy, cov_xy};
        } else {
            // diagonal case: just pick whichever axis has more spread
            major_axis = (cov_xx >= cov_yy)
                         ? cv::Point2d(1, 0)
                         : cv::Point2d(0, 1);
        }

        // normalize it
        double norm = std::sqrt(major_axis.x * major_axis.x
                                + major_axis.y * major_axis.y);
        if (norm > 1e-8) {
            major_axis.x /= norm;
            major_axis.y /= norm;
        }

        // project onto major axis to find the two extreme endpoints
        double min_proj =  1e30;
        double max_proj = -1e30;
        size_t min_idx = 0, max_idx = 0;

        for (size_t k = 0; k < pts.size(); ++k) {
            double proj = (pts[k].x - mean.x) * major_axis.x
                        + (pts[k].y - mean.y) * major_axis.y;
            if (proj < min_proj) { min_proj = proj; min_idx = k; }
            if (proj > max_proj) { max_proj = proj; max_idx = k; }
        }

        StreakDetection streak;
        streak.x1 = static_cast<int>(pts[min_idx].x);
        streak.y1 = static_cast<int>(pts[min_idx].y);
        streak.x2 = static_cast<int>(pts[max_idx].x);
        streak.y2 = static_cast<int>(pts[max_idx].y);
        streak.length = std::hypot(streak.x2 - streak.x1,
                                    streak.y2 - streak.y1);
        streak.num_fragments = static_cast<int>(cluster_indices.size());
        merged.push_back(streak);
    }

    return merged;
}


std::vector<StreakDetection> detect_streaks(
    const cv::Mat& image,
    const StreakDetectorConfig& config,
    std::vector<LineSegment>* raw_segments_out,
    StreakTimingInfo* timing) {

    double t_pre = 0, t_edge = 0, t_hough = 0, t_cluster = 0;

    cv::Mat smoothed, residual, edges;

    // bg subtract + bilateral smoothing (keeps edges, kills noise)
    {
        ScopedTimer timer(t_pre);
        cv::Mat bg = estimate_background(image, config.morph_kernel_size);
        cv::subtract(image, bg, residual);
        cv::bilateralFilter(residual, smoothed, 5, 40, 40);
    }

    // canny
    {
        ScopedTimer timer(t_edge);
        cv::Canny(smoothed, edges, config.canny_low, config.canny_high, 3);
    }

    // probabilistic hough -- gives us line segment endpoints directly
    std::vector<LineSegment> segments;
    {
        ScopedTimer timer(t_hough);
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI / 180,
                         config.hough_threshold,
                         config.hough_min_length,
                         config.hough_max_gap);

        segments.reserve(lines.size());
        for (const auto& l : lines) {
            LineSegment seg;
            seg.x1 = l[0]; seg.y1 = l[1];
            seg.x2 = l[2]; seg.y2 = l[3];
            seg.length = std::hypot(l[2] - l[0], l[3] - l[1]);
            seg.angle_rad = std::atan2(l[3] - l[1], l[2] - l[0]);
            segments.push_back(seg);
        }
    }

    if (raw_segments_out != nullptr) {
        *raw_segments_out = segments;
    }

    // merge co-linear fragments
    std::vector<StreakDetection> merged;
    {
        ScopedTimer timer(t_cluster);
        merged = cluster_segments(segments, config.cluster_angle_tol,
                                  config.cluster_dist_tol);
    }

    if (timing != nullptr) {
        timing->preprocess_ms = t_pre;
        timing->edge_detect_ms = t_edge;
        timing->hough_ms = t_hough;
        timing->clustering_ms = t_cluster;
        timing->total_ms = t_pre + t_edge + t_hough + t_cluster;
    }

    return merged;
}


cv::Mat add_synthetic_streaks(
    const cv::Mat& image,
    int num_streaks,
    std::vector<std::array<int, 4>>& streak_endpoints,
    unsigned int seed) {

    std::mt19937 rng(seed);
    int h = image.rows;
    int w = image.cols;
    std::uniform_int_distribution<int> rand_x(10, w - 10);
    std::uniform_int_distribution<int> rand_y(10, h - 10);
    std::uniform_real_distribution<double> rand_angle(0.0, M_PI);
    std::uniform_real_distribution<double> rand_len(80.0, 300.0);
    std::uniform_int_distribution<int> rand_bright(100, 200);

    cv::Mat result;
    image.convertTo(result, CV_8UC1);

    streak_endpoints.clear();
    for (int i = 0; i < num_streaks; ++i) {
        int x1 = rand_x(rng);
        int y1 = rand_y(rng);
        double angle = rand_angle(rng);
        double length = rand_len(rng);
        int x2 = std::clamp(static_cast<int>(x1 + length * std::cos(angle)),
                             0, w - 1);
        int y2 = std::clamp(static_cast<int>(y1 + length * std::sin(angle)),
                             0, h - 1);
        int bright = rand_bright(rng);

        streak_endpoints.push_back({x1, y1, x2, y2});
        cv::line(result, cv::Point(x1, y1), cv::Point(x2, y2),
                 cv::Scalar(bright), 2, cv::LINE_AA);
    }

    return result;
}

} // namespace sslab
