#pragma once

#include <array>
#include <cstdint>
#include <ostream>

#include "nlohmann/json.hpp"
#include "tops/coti/type_info.h"
#include "tops/coti/type_trait.h"

class msgpack_packer;
class msgpack_object;

namespace tops {
namespace coti {

class TypeInfo;
class Hasher;

nlohmann::json to_json(const TypeInfo &type_info, const void *object);

template <typename ObjT>
nlohmann::json to_json(const ObjT &object) {
  return to_json(get_type_info(object), &object);
}

void from_json(const TypeInfo &type_info, void *object,
               const nlohmann::json &json);

template <typename ObjT>
ObjT from_json(const nlohmann::json &json) {
  ObjT object;
  from_json(get_type_info(object), &object, json);
  return object;
}

void to_msgpack(const TypeInfo &type_info, const void *object,
                msgpack_packer &packer);

template <typename ObjT>
void to_msgpack(const ObjT &object, msgpack_packer &packer) {
  return to_msgpack(get_type_info(object), &object, packer);
}

// TODO: improve the implementation such that we don't have to construct a full
// msgpack_object in memory.
void from_msgpack(const TypeInfo &type_info, void *object,
                  const msgpack_object &msg_obj);

template <typename ObjT>
ObjT from_msgpack(const msgpack_object &msg_obj) {
  ObjT object;
  from_msgpack(get_type_info(object), &object, msg_obj);
  return object;
}

void hash(const TypeInfo &type_info, const void *object, Hasher &hasher);

template <typename ObjT>
void hash(const ObjT &object, Hasher &hasher) {
  return hash(get_type_info(object), &object, hasher);
}

}  // namespace coti
}  // namespace tops
