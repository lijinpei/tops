#pragma once

#include <climits>

namespace tops {

template <typename T> bool is_power_of_2(T x) { return (x & (x - 1)) == 0; }

template <typename T> T cdiv(T a, T b) { return (a + b - 1) / b; }

template <typename T> T next_power_of_2(T x) {
  return T(1) << (sizeof(T) * CHAR_BIT - __builtin_clz(x - 1));
}

} // namespace tops
