#pragma once

#include "cuda.h"

#include <utility>

namespace tops {
namespace gpu {
struct LaunchGrid {
  dim3 grid_shape;
  dim3 block_shape;
  template <typename T1, typename T2>
  LaunchGrid(T1 &&arg1, T2 &&arg2)
      : grid_shape(std::forward<T1>(arg1)),
        block_shape(std::forward<T2>(arg2)) {}
};
} // namespace gpu
} // namespace tops
