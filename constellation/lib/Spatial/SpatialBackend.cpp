//
// Created by Nathan Zhang on 4/25/19.
//

#include "constellation/Spatial/SpatialBackend.h"

namespace constellation {
namespace spatial {

std::string indentString(std::string s, int indent) {
    std::regex replacement_regex("\n");
    return std::regex_replace(s, replacement_regex, "\n" + std::string(indent, ' '));
}

std::string toSpatialType(mlir::Type type) {
    if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
        return "FixPt[TRUE, _" + std::to_string(intType.getWidth()) + ", _0]";
    }

    llvm_unreachable("Invalid type passed to toSpatialType");
}

void SpatialFunc::emit(llvm::raw_ostream *os) {
    nlohmann::json sections = {
            {"preAccel",  "pre"},
            {"accel",     "inside"},
            {"postAccel", "post"}
    };
    sections["name"] = func_->getName().str();

    // Create file reads
    nlohmann::json argData;
    argData["args"] = nlohmann::json::array();
    for (auto arg : func_->getArguments()) {
        nlohmann::json argInfo;
        mlir::Type argType = arg->getType();
        argInfo["typestr"] = "T";
        if (argType.isa<mlir::VectorOrTensorType>()) {
            argInfo["shape"] = nlohmann::json::array();
            for (auto dim : argType.cast<mlir::VectorOrTensorType>().getShape()) {
                argInfo["shape"].push_back(dim);
            }
        }
        argData["args"].push_back(argInfo);
    }


    sections["args"] = indentString(inja::render(templates::kArgTemplate, argData), 8);

    *os << inja::render(templates::kTopTemplate, sections);
}
}
}
