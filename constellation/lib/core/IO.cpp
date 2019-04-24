//
// Created by Nathan Zhang on 4/19/19.
//

#include "constellation/core/IO.h"

namespace constellation {
    namespace IO {
        void ReadOp::build(mlir::Builder *b, mlir::OperationState *state, std::string url,
                           constellation::IO::AccessMode accessMode, mlir::Type stateType) {
            // The type of the result should match the type of the file, but isn't checkable since the file doesn't
            // necessarily have to exist.
            setAccessMode(b, state, accessMode);
            state->addAttribute("url", b->getStringAttr(url));
            state->addTypes({stateType});
        }

        mlir::LogicalResult ReadOp::verify() {
            // Just checking if access mode is valid.
            getAccessMode();
            return mlir::success();
        }

//        bool ReadOp::parse(mlir::OpAsmParser *parser, mlir::OperationState *state) {
//            llvm_unreachable("Parse not implemented for ReadOp");
//        }

        void WriteOp::build(mlir::Builder *b, mlir::OperationState *state, std::string url,
                            constellation::IO::AccessMode accessMode, mlir::Value *data) {
            setAccessMode(b, state, accessMode);
            state->addAttribute("url", b->getStringAttr(url));
            state->addOperands({data});
            state->addTypes({b->getIntegerType(8)});
        }

        mlir::LogicalResult WriteOp::verify() {
            getAccessMode();
            return mlir::success();
        }

//        bool WriteOp::parse(mlir::OpAsmParser *parser, mlir::OperationState *state) {
//            llvm_unreachable("Parse not implemented for WriteOp");
//        }
    }
}
