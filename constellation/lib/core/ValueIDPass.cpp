//
// Created by Nathan Zhang on 2019-05-08.
//

#include "constellation/core/ValueIDPass.h"

namespace constellation::core::passes {
    std::string ValueIDPass::kAttributeName = "ValueID";

    int ValueIDPass::getID(mlir::Operation *op) {
        if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(kAttributeName)) {
            return attr.getInt();
        }
        llvm_unreachable("Provided operation has no ValueID");
    }

    void ValueIDPass::runOnFunction() {
        auto& func = getFunction();
        int nextID = func.getNumArguments();
        func.walk([&func, &nextID](mlir::Operation* op){
            op->setAttr(kAttributeName, mlir::IntegerAttr::get(
                    mlir::IntegerType::get(sizeof(int)*8, func.getContext()), nextID++));
        });
    }
}
