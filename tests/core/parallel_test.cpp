#include "colite/core/parallel.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace colite {
namespace {

TEST(ParallelRuntimeTest, PreservesThreadConfigurationAndCompletesWork) {
  parallel::setNumThreads(1);
  EXPECT_EQ(parallel::getNumThreads(), 1);

  const int threads = static_cast<int>(
      std::max(1u, std::thread::hardware_concurrency()));
  parallel::setNumThreads(threads);
  EXPECT_EQ(parallel::getNumThreads(), threads);

  for (int iteration = 0; iteration < 20; ++iteration) {
    std::vector<int> values(10000, 0);
    parallel::parallel_for_(
        parallel::Range(0, static_cast<int>(values.size())),
        [&](const parallel::Range& range) {
          for (int i = range.start; i < range.end; ++i) {
            values[static_cast<std::size_t>(i)] = i + iteration;
          }
        });
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
      ASSERT_EQ(values[static_cast<std::size_t>(i)], i + iteration);
    }
  }
}

TEST(ParallelRuntimeTest, SupportsIndependentPerRangeAllocations) {
  std::vector<std::vector<int>> rows(1024);
  parallel::parallel_for_(
      parallel::Range(0, static_cast<int>(rows.size())),
      [&](const parallel::Range& range) {
        for (int row = range.start; row < range.end; ++row) {
          for (int column = 0; column < 16; ++column) {
            rows[static_cast<std::size_t>(row)].push_back(row * 16 + column);
          }
        }
      });

  for (const auto& row : rows) EXPECT_EQ(row.size(), std::size_t{16});
}

}  // namespace
}  // namespace colite
