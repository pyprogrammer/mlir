//
// Created by Nathan Zhang on 4/30/19.
//

#ifndef CONSTELLATION_TYPES_H
#define CONSTELLATION_TYPES_H

#include <map>

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/StandardTypes.h"

namespace constellation {

    enum ConstellationTypeKind {
        CONST_TYPE = mlir::Type::FIRST_CONSTELLATION_TYPE,
        CONST_FIX, CONST_FLOAT
    };

    namespace detail {
        using KeyTy = std::tuple<bool, unsigned, unsigned>;

        template<ConstellationTypeKind CKind>
        struct ConstNumTypeStorage : public mlir::TypeStorage {
            using KeyTy = detail::KeyTy;

            // Type, sign, pre-decimal, post-decimal
            KeyTy data;

            explicit ConstNumTypeStorage(KeyTy key) : data(key) {}

            KeyTy getKey() const {
                return data;
            };

            static llvm::hash_code hashKey(const KeyTy &key) {
                return llvm::hash_combine(
                        llvm::hash_value(static_cast<std::underlying_type<ConstellationTypeKind>::type>(CKind)),
                        llvm::hash_value(std::get<0>(key)),
                        llvm::hash_value(std::get<1>(key)),
                        llvm::hash_value(std::get<2>(key))
                        );
            }

            bool operator==(const KeyTy &key) const {
                return key == getKey();
            }

            static ConstNumTypeStorage<CKind> *construct(mlir::TypeStorageAllocator &allocator,
                                                 KeyTy key) {
                return new (allocator.allocate<ConstNumTypeStorage<CKind>>())
                        ConstNumTypeStorage<CKind>(key);
            }
        };
    }

    template<ConstellationTypeKind CKind>
    struct ConstNum : public mlir::Type::TypeBase<ConstNum<CKind>, mlir::Type, detail::ConstNumTypeStorage<CKind>> {
        using Base = mlir::Type::TypeBase<ConstNum<CKind>, mlir::Type, detail::ConstNumTypeStorage<CKind>>;
        using Base::Base;

        static bool kindof(unsigned kind) {
            return kind == CKind;
        }

        bool is_signed() {
            return std::get<1>(getKey());
        }

        detail::KeyTy getKey() {
            return Base::getImpl()->getKey();
        }

        static ConstNum<CKind> get(mlir::MLIRContext *context, detail::KeyTy params) {
            return Base::get(context, CKind, params);
        }
    };

    namespace impl {
        using SKind = mlir::StandardTypes::Kind;
        const static std::map<unsigned, detail::KeyTy> dataMap = {
                {SKind::F32, {true, 8, 23}},
                {SKind::F64, {true, 11, 52}}
        };
    }

    inline ConstNum<CONST_FIX> fromIntegerType(mlir::IntegerType tp) {
        return ConstNum<CONST_FIX>::get(tp.getContext(), detail::KeyTy(true, tp.getWidth() - 1, 0));
    }

    inline ConstNum<CONST_FLOAT> fromFloatType(mlir::FloatType tp) {
        return ConstNum<CONST_FLOAT>::get(tp.getContext(), impl::dataMap.at(tp.getKind()));
    }
}
#endif //CONSTELLATION_TYPES_H
