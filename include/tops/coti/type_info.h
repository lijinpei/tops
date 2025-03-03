#pragma once

/// Limitations:
/// - Each cpp object is assumed to be of fixed size.
/// - For container objects (List/Array/Dict), it's assumed that we can get a
/// pointer to each child, or key-value pair element.
/// - Limitations of MsgPack and Json also applies to TypeKind.
/// 	* Dict key has to be string, due to json.

#include <climits>
#include <cstdint>
#include <string_view>

namespace tops {
namespace coti {
class Hasher;

enum TypeKind : uint8_t {
  // MsgPack boolean; Json true/false.
  OK_Bool,
  // An int64_t int.
  // MsgPack integer; Json number without fraction/exponent.
  OK_Int,
  // Double.
  // MsgPack float; Json number with fraction/exponent.
  OK_Float,
  // Bytes array.
  // MsgPack raw-binary; Json string.
  OK_String,
  // Homogeneous Array, variable length.
  // MsgPack array; Json array.
  OK_Array,
  // Heterogeneous Array, variable length.
  // MsgPack array; Json array.
  OK_List,
  // Key-value map.
  // MsgPack map; Json object.
  OK_Dict,
};

struct TypeInfo {
  TypeKind kind;
  // The byte-size of a cpp object. Not related to the byte-size after
  // serialization.
  unsigned cppByteSize;

  TypeInfo(TypeKind kind, unsigned cppByteSize)
      : kind(kind), cppByteSize(cppByteSize) {}

  virtual ~TypeInfo();
  void hash(const void *object, Hasher &hasher);
};

struct BoolInfo : TypeInfo {
  BoolInfo() : TypeInfo(OK_Bool, 1) {}

  bool get(const void *object) const {
    return *reinterpret_cast<const bool *>(object);
  }

  bool get(const bool &object) const { return object; }

  void set(void *object, bool value) const {
    *reinterpret_cast<bool *>(object) = value;
  }

  void set(bool &object, bool value) const { object = value; }

  using JsonValType = bool;
  static const BoolInfo &getSingleton();
};

struct IntegerInfo : TypeInfo {
  static IntegerInfo &getSBitIntInfo(uint32_t bitWidth);
  static IntegerInfo &getUBitIntInfo(uint32_t bitWidth);

  struct BitWidthAndSign {
    uint32_t bitWidth;
    bool isSigned;
  };

  bool isSigned() const { return decodeSignBit(signBitEncode).isSigned; }

  uint32_t getBitWidth() const { return decodeSignBit(signBitEncode).bitWidth; }

  int64_t get(const void *object) const;

  void set(void *object, int64_t val) const;

  using JsonValType = int64_t;

  using SignBitEncodeTy = uint32_t;
  static SignBitEncodeTy encodeSignBit(bool isSigned, uint32_t bitWidth);
  static BitWidthAndSign decodeSignBit(SignBitEncodeTy signAndBit);

 private:
  SignBitEncodeTy signBitEncode;

  IntegerInfo(bool isSigned, uint32_t bitWidth)
      : TypeInfo(OK_Int, bitWidth / CHAR_BIT),
        signBitEncode(encodeSignBit(isSigned, bitWidth)) {}
};

struct FloatInfo : TypeInfo {
  enum FloatKind : uint8_t {
    IEEE,
    BFloat,
  };

  struct BitWidthAndKind {
    uint32_t bitWidth;
    FloatKind kind;
  };

  FloatInfo(uint32_t bitWidth, FloatKind kind = IEEE)
      : TypeInfo(OK_Float, bitWidth / CHAR_BIT),
        kindBitEncode(encodeBitKind(bitWidth, kind)) {}

  static FloatInfo &getSingleton(uint32_t bitWidth, FloatKind kind);

  uint32_t getBitWidth() const { return decodeBitKind(kindBitEncode).bitWidth; }

  FloatKind getKind() const { return decodeBitKind(kindBitEncode).kind; }

  double get(const void *object) const;

  void set(void *object, double val) const;

  using JsonValType = double;

  uint64_t getU64ZExt(const void *object) const;

  void setU64(void *object, uint64_t val) const;

  using BitKindEncodeTy = uint32_t;
  static BitKindEncodeTy encodeBitKind(uint32_t bitWidth, FloatKind kind);
  static BitWidthAndKind decodeBitKind(BitKindEncodeTy encode);

 private:
  BitKindEncodeTy kindBitEncode;
};

struct StringInfo : TypeInfo {
  StringInfo(uint32_t cppByteSize) : TypeInfo(OK_String, cppByteSize) {}

  virtual std::string_view get(const void *object) const = 0;

  virtual void set(void *object, std::string_view val) const = 0;

  using JsonValType = std::string_view;
};

// LLimitation: child elements are assumed to have contiguous storage.
struct ArrayInfo : TypeInfo {
  ArrayInfo(uint32_t cppByteSize, const TypeInfo &childInfo)
      : TypeInfo(OK_Array, cppByteSize), childInfo(childInfo) {}

  virtual size_t getNumChilds(const void *object) const = 0;
  virtual void resize(const void *object, size_t newNumChild) const = 0;

  virtual const void *getChildBegin(const void *object) const = 0;

  void *getChildBegin(void *object) const {
    return const_cast<void *>(getChildBegin(object));
  }

  const TypeInfo &childInfo;
};

struct TypedPtr {
  void *ptr;
  const TypeInfo *type_info;
};

struct ListInfo : TypeInfo {
  ListInfo(uint32_t cppByteSize) : TypeInfo(OK_List, cppByteSize) {}

  virtual size_t getNumChilds(const void *object) const = 0;
  virtual void resize(const void *object, size_t newNumChild) const = 0;

  virtual TypedPtr getChildAt(const void *object, size_t childIdx) const = 0;
};

struct DictInfo : TypeInfo {
  DictInfo(uint32_t cppByteSize) : TypeInfo(OK_Dict, cppByteSize) {}

  virtual size_t getNumItems(const void *object) const = 0;

  virtual const void *beginIter(const void *object) const = 0;

  virtual bool isEndIter(const void *object, const void *iter) const = 0;

  virtual const void *nextIter(const void *object, const void *iter) const = 0;

  virtual std::string_view getKeyAtIter(const void *object,
                                        const void *iter) const = 0;

  virtual TypedPtr getValueAtIter(const void *object,
                                  const void *iter) const = 0;

  virtual void finishIter(const void *object, const void *iter) const = 0;

  virtual TypedPtr getValueAt(const void *object,
                              std::string_view key) const = 0;

  virtual void eraseValueAt(const void *object, std::string_view key) const = 0;
};
}  // namespace coti
}  // namespace tops
