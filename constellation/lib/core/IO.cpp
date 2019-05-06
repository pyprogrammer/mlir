//
// Created by Nathan Zhang on 4/19/19.
//

#include "constellation/core/IO.h"

namespace constellation::IO {
    void ReadOp::build(mlir::Builder *b, mlir::OperationState *state, std::string url,
                       AccessMode accessMode, mlir::Type stateType) {
        // The type of the result should match the type of the file, but isn't checkable since the file doesn't
        // necessarily have to exist.
        setEnumAttr<0>(b, state, accessMode);
        state->addAttribute("url", b->getStringAttr(url));
        state->addTypes({stateType});
    }

    mlir::LogicalResult ReadOp::verify() {
        // Just checking if access mode is valid.
        return mlir::success();
    }

    void WriteOp::build(mlir::Builder *b, mlir::OperationState *state, std::string url,
                        AccessMode accessMode, mlir::Value *data) {
        setEnumAttr<0>(b, state, accessMode);
        state->addAttribute("url", b->getStringAttr(url));
        state->addOperands({data});
        state->addTypes({b->getIntegerType(8)});
    }

    mlir::LogicalResult WriteOp::verify() {
        return mlir::success();
    }

    void TransferOp::build(mlir::Builder *b, mlir::OperationState *state, AccessMode accessMode, mlir::Value *data,
            Memory input, Memory output) {
        setEnumAttr<0>(b, state, accessMode);
        state->addTypes({data->getType()});
        state->addOperands({data});
        state->addAttribute("input", b->getIntegerAttr(getMemorySpaceType(b), Memory::MLIRMemorySpaceType(input)));
        state->addAttribute("output", b->getIntegerAttr(getMemorySpaceType(b), Memory::MLIRMemorySpaceType(output)));
    }

    Memory TransferOp::getInputMem() {
        return Memory(getAttrOfType<mlir::IntegerAttr>("input").getInt());
    }

    Memory TransferOp::getOutputMem() {
        return Memory(getAttrOfType<mlir::IntegerAttr>("output").getInt());
    }

    mlir::LogicalResult TransferOp::verify() {
        if (getResult()->getType() != getType()) {
            return mlir::failure();
        }
        return mlir::success();
    }
}
