#pragma once

#include <array>

namespace tops {
namespace coti {
using SHA3_256DigestTy = std::array<unsigned char, 64>;
class Hasher;
// namespace {
// class SHA3_256Hasher {
//   OSSL_LIB_CTX *ctx;
//   EVP_MD *md;
//   EVP_MD_CTX *mdctx;
//
// public:
//   SHA3_256Hasher() {
//     ctx = OSSL_LIB_CTX_new();
//     md = EVP_MD_fetch(ctx, "SHA3-256", nullptr);
//     assert(md);
//     mdctx = EVP_MD_CTX_new();
//   }
//
//   void reinit() {
//     auto res = EVP_DigestInit_ex2(mdctx, md, nullptr);
//     (void)res;
//     assert(res);
//   }
//
//   template <typename T> void update(const T &val) {
//     update((const char *)&val, sizeof(T));
//   }
//
//   void update(const char *data, size_t cnt) {
//     auto res = EVP_DigestUpdate(mdctx, data, cnt);
//     (void)res;
//     assert(res);
//   }
//
//   SHA3_256DigestTy finalize() {
//     SHA3_256DigestTy digest;
//     unsigned mdLen;
//     auto res = EVP_DigestFinal_ex(mdctx, digest.data(), &mdLen);
//     (void)mdLen;
//     assert(mdLen == digest.size());
//     return digest;
//   }
//
//   ~SHA3_256Hasher() {
//     EVP_MD_CTX_free(mdctx);
//     EVP_MD_free(md);
//     OSSL_LIB_CTX_free(ctx);
//   }
// };
//
// void hash(const TypeInfo &type_info, const char *object,
//           SHA3_256Hasher &hasher);
// void hash_array(const ArrayInfo &array_info, const char *object,
//                 SHA3_256Hasher &hasher);
//
// template <typename T> struct Hash {
//   static void run(const T &type_info, const char *object,
//                   SHA3_256Hasher &hasher) {
//     if constexpr (std::is_same_v<T, BoolInfo>) {
//       char val = type_info.getVal(object) ? '1' : '0';
//       hasher.update(val);
//     }
//     if constexpr (std::is_same_v<T, IntegerInfo>) {
//       if (type_info.isSigned()) {
//         int64_t val = type_info.getSignedVal(object);
//         hasher.update(val);
//       } else {
//         uint64_t val = type_info.getUnsignedVal(object);
//         hasher.update(val);
//       }
//     }
//     if constexpr (std::is_same_v<T, FloatInfo>) {
//       double val = type_info.getVal(object);
//       hasher.update(val);
//     }
//     if constexpr (std::is_same_v<T, StringInfo>) {
//       auto val = type_info.getVal(object);
//       hasher.update(val.data(), val.size());
//     }
//     if constexpr (std::is_same_v<T, ArrayInfo>) {
//       hash_array(type_info, object, hasher);
//     }
//     if constexpr (std::is_same_v<T, DictInfo>) {
//       for (uint32_t i = 0; i < type_info.numChilds; ++i) {
//         auto name = type_info.names[i];
//         hasher.update(name.data(), name.size());
//         hash(type_info.elems[i], object + type_info.offsets[i], hasher);
//       }
//     }
//   }
// };
//
// void hash(const TypeInfo &type_info, const char *object,
//           SHA3_256Hasher &hasher) {
//   return type_info_dispatch<Hash>(type_info, object, hasher);
// }
//
// template <typename T> struct HashArray {
//   static void run(const T &type_info, const ArrayInfo &array_info,
//                   const char *object, SHA3_256Hasher &hasher) {
//     uint32_t size = array_info.size;
//     hasher.update(size);
//     uint32_t elemSize = type_info.sizeInBytes;
//     for (uint32_t idx = 0; idx < size; ++idx) {
//       Hash<T>::run(type_info, object + idx * elemSize, hasher);
//     }
//   }
// };
//
// void hash_array(const ArrayInfo &array_info, const char *object,
//                 SHA3_256Hasher &hasher) {
//   return type_info_dispatch<HashArray>(array_info.elem, array_info, object,
//                                        hasher);
// }
//
// thread_local SHA3_256Hasher hasher;
//
// } // namespace
//
//
// SHA3_256DigestTy hash(const TypeInfo &type_info, const void *object) {
//   hasher.reinit();
//   hash(type_info, object, hasher);
//   return hasher.finalize();

}  // namespace coti
}  // namespace tops
