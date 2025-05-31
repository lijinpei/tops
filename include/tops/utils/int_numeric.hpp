#pragma once

namespace tops {

template <typename T> bool is_power_of_2(T x) { return (x & (x - 1)) == 0; }

template <typename T> T cdiv(T a, T b) { return (a + b - 1) / b; }

} // namespace tops
