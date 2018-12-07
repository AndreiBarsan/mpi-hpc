/*
 * @file mpi_stopwatch.h
 * @brief Defines a helper class which can time code running with MPI.
 */

#ifndef HPSC_MPISTOPWATCH_H
#define HPSC_MPISTOPWATCH_H

#include <chrono>
#include <map>

#include "common/mpi_helpers.h"
#include "common/utils.h"

class MPIStopwatch {
 public:
  void Start() {
    last_time_ = std::chrono::system_clock::now();
  }

  void Record(const std::string &label) {
    auto now = std::chrono::system_clock::now();
    if (records_.find(label) != records_.end()) {
      throw std::runtime_error(Format("Double-record of label %s!", label.c_str()));
    }

    records_[label] = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time_);
    last_time_ = now;
  }

  Duration GetMaxTimeUs(const std::string &label) const {
    MPI_SETUP;
    long time_us = std::chrono::microseconds(records_.at(label)).count();
    long max_time_us = -1;
    MPI_Allreduce(&time_us, &max_time_us, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    assert (max_time_us >= 0);
    return Duration(std::chrono::microseconds(max_time_us));
  }

  Duration GetMaxTotalTimeUs() const {
    long time_us = 0.0;
    for (const auto& p : records_) {
      time_us += std::chrono::microseconds(p.second).count();
    }
    long max_time_us = -1L;
    MPI_Allreduce(&time_us, &max_time_us, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    assert (max_time_us >= 0);
    return Duration(std::chrono::microseconds(max_time_us));
  }

  std::map<std::string, Duration> GetAllMaxTimesUs() const {
    std::map<std::string, Duration> max_records;
    for(const auto &entry : records_) {
      max_records[entry.first] = GetMaxTimeUs(entry.first);
    }
    return max_records;
  }

 private:
  std::chrono::system_clock::time_point last_time_;
  std::map<std::string, Duration> records_;
};

#endif //HPSC_MPISTOPWATCH_H
