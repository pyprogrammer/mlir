//
// Created by Nathan Zhang on 4/25/19.
//

#include <constellation/core/Lattice.h>
#include "constellation/Spatial/SpatialBackend.h"

namespace constellation::spatial {

namespace detail {
    std::string indentString(std::string s, int indent) {
        std::regex replacement_regex("\n");
        return std::regex_replace(s, replacement_regex, "\n" + std::string(indent, ' '));
    }

    template<ConstellationTypeKind CKind>
    std::string constellationTypeToString(ConstNum<CKind> num) {
        static const std::map<ConstellationTypeKind, std::string> typePrefixes = {
                {CONST_FIX, "FixPt"},
                {CONST_FLOAT, "FltPt"}
        };

        // Spatial numbers include bit in the int/exp count, so we need to increase by 1.
        bool is_signed;
        int first;
        int second;
        std::tie(is_signed, first, second) = num.getKey();
        std::string sign = is_signed ? "TRUE" : "FALSE";
        return typePrefixes.at(CKind) + "[" + sign + ", _" + std::to_string(first) + ", _" + std::to_string(second) + "]";
    }
}

void SpatialFunc::initPassManager(mlir::PassManager *pm) {
    pm->addPass(new passes::SpatialLocationPass());
}

void SpatialFunc::emit(llvm::raw_ostream *os) {

    func_->print(*os);

    nlohmann::json sections = {
            {"host",  "host"},
            {"accel", "inside"}
    };
    sections["name"] = func_->getName().str();

    // Create file reads
    nlohmann::json argData;
    argData["args"] = nlohmann::json::array();
    for (auto arg : func_->getArguments()) {
        nlohmann::json argInfo;
        mlir::Type argType = arg->getType();
//        argInfo["typestr"] = "T";
        if (auto vecType = argType.dyn_cast<mlir::VectorOrTensorType>()) {
            argInfo["shape"] = nlohmann::json::array();
            for (auto dim : vecType.getShape()) {
                argInfo["shape"].push_back(dim);
            }

            if (auto floatType = vecType.getElementType().dyn_cast<mlir::FloatType>()) {
                argInfo["typestr"] = detail::constellationTypeToString(fromFloatType(floatType));
            } else if (auto intType = vecType.getElementType().dyn_cast<mlir::IntegerType>()) {
                argInfo["typestr"] = detail::constellationTypeToString(fromIntegerType(intType));
            } else {
                llvm_unreachable("Don't know how to convert type");
            }
        }
        argData["args"].push_back(argInfo);
    }


    sections["args"] = detail::indentString(inja::render(templates::kArgTemplate, argData), 8);

    *os << inja::render(templates::kTopTemplate, sections);
}
}
