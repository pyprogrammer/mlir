//
// Created by Nathan Zhang on 4/19/19.
//

#ifndef LLVM_CONSTELLATION_ENUMUTILS_H
#define LLVM_CONSTELLATION_ENUMUTILS_H

#include <type_traits>

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

namespace constellation {
    template<typename T>
    mlir::IntegerType toIntegerType(mlir::Builder *builder) {
        static_assert(std::is_enum<T>::value, "Only Enums are allowed for conversion to MLIR IntegerType");
        return builder->getIntegerType(sizeof(typename std::underlying_type<T>::type) * 8);
    }

    template<typename T>
    mlir::IntegerAttr toIntegerAttr(mlir::Builder *builder, T val) {
        return builder->getIntegerAttr(
                toIntegerType<T>(builder),
                static_cast<typename std::underlying_type<T>::type>(val)
        );
    }

    template<typename T>
    T fromIntegerAttr(mlir::IntegerAttr integerAttr) {
        static auto last = static_cast<typename std::underlying_type<T>::type>(T::ENUM_LAST);
        auto value = integerAttr.getInt();
        assert(value >= 0 && "Value provided was less than 0");
        assert(value <= last && "Value provided was greater than enum range.");
        return static_cast<T>(value);
    }
}

#endif //LLVM_CONSTELLATION_ENUMUTILS_H
