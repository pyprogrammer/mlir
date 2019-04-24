//
// Created by Nathan Zhang on 4/19/19.
//

#include "constellation/IO.h"

namespace constellation {
    namespace IO {
        void ReadOp::build(mlir::Builder *b, mlir::OperationState *result, std::string url,
                           constellation::IO::AccessMode accessMode, mlir::Type resultType) {
            // The type of the result should match the type of the file, but isn't checkable since the file doesn't
            // necessarily have to exist.
            result->addTypes({resultType});
            setAccessMode(b, result, accessMode);

            result->addAttribute("url", b->getStringAttr(url));
        }

        mlir::LogicalResult ReadOp::verify() {
            // Just checking if access mode is valid.
            getAccessMode();
            return mlir::success();
        }

        bool ReadOp::parse(mlir::OpAsmParser *parser, mlir::OperationState *result) {
            llvm_unreachable("Parse not implemented for ReadOp");
        }
    }
}
