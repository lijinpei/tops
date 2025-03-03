#pragma once

#include <array>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "tops/coti/type_info.h"

namespace tops {
namespace coti {
namespace impl {
const StringInfo& getStdStrigTypeInfo();
const StringInfo& getStdStrigViewTypeInfo();
const ArrayInfo& getStdVectorTypeInfo();
const ArrayInfo& getStdArrayTypeInfo();
}  // namespace impl

template <typename ObjT, typename = void>
struct TypeTrait;

template <typename ObjT>
const TypeInfo& get_type_info() {
  return TypeTrait<ObjT>::type_info();
}

template <typename ObjT>
const TypeInfo& get_type_info(const ObjT&) {
  return get_type_info<ObjT>();
}

template <>
struct TypeTrait<bool> {
  const BoolInfo& type_info() { return BoolInfo::getSingleton(); }
};

template <typename T>
struct TypeTrait<T, std::enable_if_t<std::is_integral_v<T>>> {
  const IntegerInfo& type_info() {
    if constexpr (std::is_signed_v<T>) {
      return IntegerInfo::getSBitIntInfo(sizeof(T) * CHAR_BIT);
    } else {
      return IntegerInfo::getUBitIntInfo(sizeof(T) * CHAR_BIT);
    }
  }
};

template <typename T>
struct TypeTrait<T, std::enable_if_t<std::is_same_v<T, double> ||
                                     std::is_same_v<T, float>>> {
  const FloatInfo& type_info() {
    return FloatInfo::getSingleton(sizeof(T) * CHAR_BIT, FloatInfo::IEEE);
  }
};

template <>
struct TypeTrait<std::string> {
  const StringInfo& type_info() { return impl::getStdStrigTypeInfo(); }
};

template <>
struct TypeTrait<std::string_view> {
  const StringInfo& type_info() { return impl::getStdStrigViewTypeInfo(); }
};

template <typename T>
struct TypeTrait<std::vector<T>> {
  const ArrayInfo& type_info() { return impl::getStdVectorTypeInfo(); }
};

template <typename T, size_t N>
struct TypeTrait<std::array<T, N>> {
  const ArrayInfo& type_info() { return impl::getStdArrayTypeInfo(); }
};

}  // namespace coti
}  // namespace tops
