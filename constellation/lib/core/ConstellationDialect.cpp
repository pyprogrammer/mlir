//
// Created by Nathan Zhang on 4/10/19.
//

#include "constellation/core/ConstellationDialect.h"

namespace constellation {
    ConstellationDialect::ConstellationDialect(mlir::MLIRContext *ctx) : mlir::Dialect("constellation", ctx) {
        addOperations<lattice::LatticeOp, IO::ReadOp, IO::WriteOp, IO::TransferOp>();
        addTypes<ConstNum<CONST_FIX>, ConstNum<CONST_FLOAT>>();
    };

    mlir::Type ConstellationDialect::parseType(llvm::StringRef spec, mlir::Location loc) const {
        llvm_unreachable("Unhandled linalg dialect parsing");
        return mlir::Type();
    }

    void ConstellationDialect::printType(mlir::Type type, llvm::raw_ostream &os) const {

    }
}
