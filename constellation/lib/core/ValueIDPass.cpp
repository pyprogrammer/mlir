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

    int ValueIDPass::getID(mlir::Value *value) {
        if (value->getKind() == mlir::Value::Kind::BlockArgument) {
            return reinterpret_cast<mlir::BlockArgument*>(value)->getArgNumber();
        }
        else if (auto op = value->getDefiningOp()) {
            return getID(op);
        }

        llvm_unreachable("Free floating value");
    }

    void ValueIDPass::setID(mlir::Operation *op, int id) {
        op->setAttr(kAttributeName, mlir::IntegerAttr::get(
                mlir::IntegerType::get(sizeof(int)*8, op->getContext()), id));
    }

    void ValueIDPass::runOnFunction() {
        auto& func = getFunction();
        int nextID = func.getNumArguments();
        func.walk([&nextID](mlir::Operation* op){
            setID(op, nextID++);
        });
    }
}
