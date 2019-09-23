//
// Created by Nathan Zhang on 2019-05-07.
//

#include "constellation/Spatial/SpatialEmitSupport.h"

namespace constellation::spatial::detail {
    std::string indentString(std::string s, int indent) {
        std::regex replacement_regex("\n");
        return std::regex_replace(s, replacement_regex, "\n" + std::string(indent, ' '));
    }

    template<>
    std::string constellationTypeToString(ConstNum<CONST_FIX> num) {
        bool is_signed;
        int first;
        int second;
        std::tie(is_signed, first, second) = num.getKey();
        std::string sign = is_signed ? "TRUE" : "FALSE";
        return "FixPt[" + sign + ", _" + std::to_string(first) + ", _" + std::to_string(second) + "]";
    }

    template<>
    std::string constellationTypeToString(ConstNum<CONST_FLOAT> num) {
        bool is_signed;
        int first;
        int second;
        std::tie(is_signed, first, second) = num.getKey();
        return "FltPt[_" + std::to_string(first) + ", _" + std::to_string(second) + "]";
    }

    std::string typeToString(mlir::Type tp) {
        if (auto fltType = tp.dyn_cast<ConstNum<CONST_FLOAT>>()) {
            return constellationTypeToString(fltType);
        }
        if (auto fltType = tp.dyn_cast<mlir::FloatType>()) {
            return constellationTypeToString(fromFloatType(fltType));
        }
        if (auto intType = tp.dyn_cast<ConstNum<CONST_FIX>>()) {
            return constellationTypeToString(intType);
        }
        if (auto intType = tp.dyn_cast<mlir::IntegerType>()) {
            return constellationTypeToString(fromIntegerType(intType));
        }
        llvm_unreachable("Don't know how to convert type to string");
    }

    nlohmann::json getShape(mlir::Type tp) {
        auto shape = nlohmann::json::array();
        if (auto vecArrType = tp.dyn_cast<mlir::VectorOrTensorType>()) {
            for (auto i: vecArrType.getShape()) {
                shape.push_back(i);
            }
            return shape;
        }
        shape.push_back(1);
        return shape;
    }

}
