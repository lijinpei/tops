#pragma once

namespace tops {
struct LoadedKernelLib {
  void (*load_func)() = nullptr;
  void (*unload_func)() = nullptr;
  void *launcher_func = nullptr;
  void *dl_handle = nullptr;
  template <typename ResTy, typename... ArgTys> ResTy call(ArgTys &&...args) {
    auto *func_ptr =
        static_cast<ResTy (*)(decltype(std::forward<ArgTys>(args))...)>(
            launcher_func);
    return func_ptr(args...);
  }
};
} // namespace tops
