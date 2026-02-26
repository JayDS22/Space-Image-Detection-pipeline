/*
 * profiler.hpp
 *
 * Simple RAII timer and multi-run stats collector. Nothing fancy --
 * just wraps chrono so we don't have to write the same start/stop
 * boilerplate in every pipeline stage.
 *
 * Usage:
 *   double ms;
 *   { ScopedTimer t(ms); do_expensive_thing(); }
 *   printf("took %.2f ms\n", ms);
 */

#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace sslab {

class ScopedTimer {
public:
    explicit ScopedTimer(double& out_ms)
        : out_ms_(out_ms),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start_;
        out_ms_ = diff.count();
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    double& out_ms_;
    std::chrono::high_resolution_clock::time_point start_;
};

struct BenchmarkStats {
    double min_ms;
    double max_ms;
    double mean_ms;
    double stddev_ms;
    int    num_runs;
};

// Collects per-stage timings over many runs so we can report
// min/max/mean/std after an optimization pass.
class PipelineBenchmark {
public:
    void record(const std::string& stage_name, double elapsed_ms) {
        data_[stage_name].push_back(elapsed_ms);
    }

    BenchmarkStats get_stats(const std::string& stage_name) const {
        BenchmarkStats stats{};
        auto it = data_.find(stage_name);
        if (it == data_.end() || it->second.empty()) {
            return stats;
        }

        const auto& v = it->second;
        stats.num_runs = static_cast<int>(v.size());
        stats.min_ms = *std::min_element(v.begin(), v.end());
        stats.max_ms = *std::max_element(v.begin(), v.end());
        stats.mean_ms = std::accumulate(v.begin(), v.end(), 0.0)
                        / v.size();

        double sq_sum = 0.0;
        for (double val : v) {
            sq_sum += (val - stats.mean_ms) * (val - stats.mean_ms);
        }
        stats.stddev_ms = std::sqrt(sq_sum / v.size());

        return stats;
    }

    std::vector<std::string> stage_names() const {
        std::vector<std::string> names;
        names.reserve(data_.size());
        for (const auto& kv : data_) {
            names.push_back(kv.first);
        }
        return names;
    }

    void clear() { data_.clear(); }

private:
    std::map<std::string, std::vector<double>> data_;
};

} // namespace sslab

#endif // PROFILER_HPP
