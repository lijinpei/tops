#include "tops/coti/utils.h"

#include <cassert>

#include "fmt/ostream.h"
#include "msgpack.h"
#include "openssl/crypto.h"
#include "openssl/evp.h"

namespace tops {
namespace coti {
namespace {

template <template <typename> typename T, typename... Args>
auto type_info_dispatch(const TypeInfo &type_info, Args &&...args) {
  switch (type_info.kind) {
    default:
      break;
    case TypeKind::OK_Bool:
      return T<BoolInfo>::run(static_cast<const BoolInfo &>(type_info),
                              std::forward<Args>(args)...);
    case TypeKind::OK_Int:
      return T<IntegerInfo>::run(static_cast<const IntegerInfo &>(type_info),
                                 std::forward<Args>(args)...);
    case TypeKind::OK_Float:
      return T<FloatInfo>::run(static_cast<const FloatInfo &>(type_info),
                               std::forward<Args>(args)...);
    case TypeKind::OK_String:
      return T<StringInfo>::run(static_cast<const StringInfo &>(type_info),
                                std::forward<Args>(args)...);
    case TypeKind::OK_Array:
      return T<ArrayInfo>::run(static_cast<const ArrayInfo &>(type_info),
                               std::forward<Args>(args)...);
    case TypeKind::OK_List:
      return T<ListInfo>::run(static_cast<const ListInfo &>(type_info),
                              std::forward<Args>(args)...);
    case TypeKind::OK_Dict:
      return T<DictInfo>::run(static_cast<const DictInfo &>(type_info),
                              std::forward<Args>(args)...);
  }
  __builtin_unreachable();
}

template <typename T>
constexpr bool has_simple_get_set_v =
    std::is_same_v<T, BoolInfo> || std::is_same_v<T, IntegerInfo> ||
    std::is_same_v<T, FloatInfo> || std::is_same_v<T, StringInfo>;

nlohmann::json to_json(const ArrayInfo &arrayInfo, const void *object);
nlohmann::json to_json(const ListInfo &listInfo, const void *object);
nlohmann::json to_json(const DictInfo &dictInfo, const void *object);

template <typename T>
struct ToJson {
  static nlohmann::json run(const T &type_info, const void *object) {
    if constexpr (has_simple_get_set_v<T>) {
      return nlohmann::json(type_info.get(object));
    }
    return to_json(type_info, object);
  }
};

template <typename T>
struct ToJsonArray {
  static nlohmann::json run(const T &elemInfo, const ArrayInfo &arrayInfo,
                            const void *object) {
    nlohmann::json json(nlohmann::json::value_t::array);
    uint32_t elemSize = elemInfo.cppByteSize;
    auto childBegin =
        reinterpret_cast<const char *>(arrayInfo.getChildBegin(object));
    for (uint32_t idx = 0, numChild = arrayInfo.getNumChilds(object);
         idx < numChild; ++idx) {
      json.push_back(ToJson<T>::run(elemInfo, childBegin + elemSize * idx));
    }
    return std::move(json);
  }
};

template <typename T>
struct ToJsonList {
  static void run(const T &type_info, nlohmann::json &json, const void *child) {
    if constexpr (has_simple_get_set_v<T>) {
      json.emplace_back(type_info.get(child));
    }
    if constexpr (std::is_same_v<T, ArrayInfo> || std::is_same_v<T, ListInfo> ||
                  std::is_same_v<T, DictInfo>) {
      auto childJson = to_json(type_info, child);
      json.push_back(childJson);
    }
  }
};

nlohmann::json to_json(const ArrayInfo &arrayInfo, const void *object) {
  return type_info_dispatch<ToJsonArray>(arrayInfo.childInfo, arrayInfo,
                                         object);
}

nlohmann::json to_json(const ListInfo &listInfo, const void *object) {
  nlohmann::json json(nlohmann::json::value_t::array);
  auto numChild = listInfo.getNumChilds(object);
  for (size_t i = 0; i < numChild; ++i) {
    auto child = listInfo.getChildAt(object, i);
    type_info_dispatch<ToJsonList>(*child.type_info, json, child.ptr);
  }
  return json;
}

nlohmann::json to_json(const DictInfo &dictInfo, const void *object) {
  nlohmann::json json(nlohmann::json::value_t::object);
  auto *iter = dictInfo.beginIter(object);
  while (!dictInfo.isEndIter(object, iter)) {
    auto key = dictInfo.getKeyAtIter(object, iter);
    auto value = dictInfo.getValueAtIter(object, iter);
    json.emplace(key, to_json(*value.type_info, value.ptr));
    iter = dictInfo.nextIter(object, iter);
  }
  dictInfo.finishIter(object, iter);
  return std::move(json);
}

void from_json(const ArrayInfo &arrayInfo, void *object,
               const nlohmann::json &json);
void from_json(const ListInfo &listInfo, void *object,
               const nlohmann::json &json);
void from_json(const DictInfo &dictInfo, void *object,
               const nlohmann::json &json);

template <typename T>
struct FromJson {
  static void run(const T &type_info, const nlohmann::json &json,
                  void *object) {
    if constexpr (has_simple_get_set_v<T>) {
      return type_info.set(object, json.get<typename T::JsonValType>());
    }
    return from_json(type_info, object, json);
  }
};

template <typename T>
struct FromJsonArray {
  static void run(const T &type_info, const nlohmann::json &json,
                  char *childBegin, size_t numChilds) {
    uint32_t elemSize = type_info.cppByteSize;
    for (size_t idx = 0; idx < numChilds; ++idx) {
      FromJson<T>::run(type_info, json[idx], childBegin + elemSize * idx);
    }
  }
};

void from_json(const ArrayInfo &arrayInfo, void *object,
               const nlohmann::json &json) {
  auto numChilds = json.size();
  arrayInfo.resize(object, numChilds);
  return type_info_dispatch<FromJsonArray>(
      arrayInfo.childInfo, json,
      reinterpret_cast<char *>(arrayInfo.getChildBegin(object)), numChilds);
}

void from_json(const ListInfo &listInfo, void *object,
               const nlohmann::json &json) {
  auto numChilds = json.size();
  listInfo.resize(object, numChilds);
  for (size_t idx = 0; idx < numChilds; ++idx) {
    auto child = listInfo.getChildAt(object, idx);
    from_json(*child.type_info, child.ptr, json[idx]);
  }
}

void from_json(const DictInfo &dictInfo, void *object,
               const nlohmann::json &json) {
  for (const auto &[key, value] : json.items()) {
    auto typedPtr = dictInfo.getValueAt(object, key);
    from_json(*typedPtr.type_info, typedPtr.ptr, value);
  }
}

template <typename T>
T safe_cast(int64_t val) {
  auto res = T(val);
  assert(res == val);
  return res;
}

void to_msgpack(const TypeInfo &elemInfo, size_t numChilds,
                const void *childBegin, msgpack_packer &packer);

template <typename T>
struct ToMsgPack {
  static void run(const T &type_info, const void *object,
                  msgpack_packer &packer) {
    if constexpr (std::is_same_v<T, BoolInfo>) {
      auto val = type_info.get(object);
      msgpack_pack_char(&packer, val ? '\1' : '\0');
      return;
    }
    if constexpr (std::is_same_v<T, IntegerInfo>) {
      auto val = type_info.get(object);
      if (type_info.isSigned()) {
        switch (type_info.getBitWidth()) {
          case 8:
            msgpack_pack_int8(&packer, safe_cast<int8_t>(val));
            return;
          case 16:
            msgpack_pack_int16(&packer, safe_cast<int16_t>(val));
            return;
          case 32:
            msgpack_pack_int32(&packer, safe_cast<int32_t>(val));
            return;
          case 64:
            msgpack_pack_int64(&packer, val);
            return;
          default:
            __builtin_unreachable();
        }
      } else {
        switch (type_info.getBitWidth()) {
          case 8:
            msgpack_pack_uint8(&packer, safe_cast<uint8_t>(val));
            return;
          case 16:
            msgpack_pack_uint16(&packer, safe_cast<uint16_t>(val));
            return;
          case 32:
            msgpack_pack_uint32(&packer, safe_cast<uint32_t>(val));
            return;
          case 64:
            assert(false && "only support 64 bit int of signed type");
            return;
          default:
            __builtin_unreachable();
        }
      }
    }
    if constexpr (std::is_same_v<T, FloatInfo>) {
      auto val = type_info.get(object);
      msgpack_pack_double(&packer, val);
      return;
    }
    if constexpr (std::is_same_v<T, StringInfo>) {
      auto val = type_info.get(object);
      msgpack_pack_str_with_body(&packer, val.data(), val.size());
      return;
    }
    if constexpr (std::is_same_v<T, ArrayInfo>) {
      return to_msgpack(type_info.childInfo, type_info.getNumChilds(object),
                        type_info.getChildBegin(object), packer);
    }
    if constexpr (std::is_same_v<T, ListInfo>) {
      auto numChilds = type_info.getNumChilds(object);
      msgpack_pack_array(&packer, numChilds);
      for (size_t idx = 0; idx < numChilds; ++idx) {
        auto typedPtr = type_info.getChildAt(object, idx);
        to_msgpack(*typedPtr.type_info, typedPtr.ptr, packer);
      }
      return;
    }
    if constexpr (std::is_same_v<T, DictInfo>) {
      auto numItems = type_info.getNumItems(object);
      msgpack_pack_map(&packer, numItems);
      auto iter = type_info.beginIter(object);
      while (!type_info.isEndIter(object, iter)) {
        auto key = type_info.getKeyAtIter(object, iter);
        msgpack_pack_str_with_body(&packer, key.data(), key.size());
        auto value = type_info.getValueAtIter(object, iter);
        to_msgpack(*value.type_info, value.ptr, packer);
        iter = type_info.nextIter(object, iter);
      }
      type_info.finishIter(object, iter);
    }
  }
};

template <typename T>
struct ToMsgPackArray {
  static void run(const T &type_info, size_t numChilds, const void *childBegin,
                  msgpack_packer &packer) {
    auto *childPtr = reinterpret_cast<const char *>(childBegin);
    auto sizeInBytes = type_info.cppByteSize;
    for (size_t idx = 0; idx < numChilds; ++idx) {
      ToMsgPack<T>::run(type_info, childPtr, packer);
      childPtr += sizeInBytes;
    }
  }
};

void to_msgpack(const TypeInfo &elemInfo, size_t numChilds,
                const void *childBegin, msgpack_packer &packer) {
  msgpack_pack_array(&packer, numChilds);
  return type_info_dispatch<ToMsgPackArray>(elemInfo, numChilds, childBegin,
                                            packer);
}

void from_msgpack(const ArrayInfo &arrayInfo, void *object,
                  msgpack_object_array array);

template <typename T>
struct FromMsgPack {
  static void run(const T &type_info, void *object,
                  const msgpack_object &msg_obj) {
    if constexpr (std::is_same_v<T, BoolInfo>) {
      assert(msg_obj.type == MSGPACK_OBJECT_BOOLEAN);
      type_info.set(object, msg_obj.via.boolean);
      return;
    }
    if constexpr (std::is_same_v<T, IntegerInfo>) {
      assert(msg_obj.type == MSGPACK_OBJECT_POSITIVE_INTEGER ||
             msg_obj.type == MSGPACK_OBJECT_NEGATIVE_INTEGER);
      type_info.set(object, msg_obj.via.i64);
      return;
    }
    if constexpr (std::is_same_v<T, FloatInfo>) {
      assert(msg_obj.type == MSGPACK_OBJECT_FLOAT);
      type_info.set(object, msg_obj.via.f64);
      return;
    }
    if constexpr (std::is_same_v<T, StringInfo>) {
      assert(msg_obj.type == MSGPACK_OBJECT_STR);
      auto str = msg_obj.via.str;
      type_info.set(object, std::string_view(str.ptr, str.size));
      return;
    }
    if constexpr (std::is_same_v<T, ArrayInfo>) {
      return from_msgpack(type_info, object, msg_obj.via.array);
    }
    if constexpr (std::is_same_v<T, ListInfo>) {
      auto array = msg_obj.via.array;
      auto size = array.size;
      type_info.resize(object, size);
      for (size_t idx = 0; idx < size; ++idx) {
        auto childPtr = type_info.getChildAt(object, idx);
        from_msgpack(*childPtr.type_info, childPtr.ptr, array.ptr[idx]);
      }
      return;
    }
    if constexpr (std::is_same_v<T, DictInfo>) {
      auto map = msg_obj.via.map;
      auto size = map.size;
      for (size_t idx = 0; idx < size; ++idx) {
        auto keyVal = map.ptr[idx];
        assert(keyVal.key.type == MSGPACK_OBJECT_STR);
        auto strObj = keyVal.key.via.str;
        auto key = std::string_view(strObj.ptr, strObj.size);
        auto valPtr = type_info.getValueAt(object, key);
        from_msgpack(*valPtr.type_info, valPtr.ptr, keyVal.val);
      }
      return;
    }
  }
};

template <typename T>
struct FromMsgPackArray {
  static void run(const T &type_info, void *childBegin, size_t numChilds,
                  msgpack_object_array array) {
    auto *child = reinterpret_cast<char *>(childBegin);
    auto sizeInBytes = type_info.cppByteSize;
    for (size_t idx = 0; idx < numChilds; ++idx) {
      FromMsgPack<T>::run(type_info, child, array.ptr[idx]);
      child += sizeInBytes;
    }
  }
};

void from_msgpack(const ArrayInfo &arrayInfo, void *object,
                  msgpack_object_array array) {
  auto size = array.size;
  arrayInfo.resize(object, size);
  return type_info_dispatch<FromMsgPackArray>(
      arrayInfo.childInfo, arrayInfo.getChildBegin(object), size, array);
}

}  // namespace

nlohmann::json to_json(const TypeInfo &type_info, const void *object) {
  return type_info_dispatch<ToJson>(type_info, object);
}

void from_json(const TypeInfo &type_info, void *object,
               const nlohmann::json &json) {
  return type_info_dispatch<FromJson>(type_info, json, object);
}

void to_msgpack(const TypeInfo &type_info, const void *object,
                msgpack_packer &packer) {
  return type_info_dispatch<ToMsgPack>(type_info, object, packer);
}

void from_msgpack(const TypeInfo &type_info, void *object,
                  const msgpack_object &msg_obj) {
  return type_info_dispatch<FromMsgPack>(type_info, object, msg_obj);
}

}  // namespace coti
}  // namespace tops
