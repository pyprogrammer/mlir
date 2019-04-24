//
// Created by Nathan Zhang on 4/17/19.
//

#ifndef LLVM_CONSTELLATION_INTRINSICS_H
#define LLVM_CONSTELLATION_INTRINSICS_H

#include "mlir/EDSC/Intrinsics.h"
#include "constellation/Lattice.h"
#include "constellation/IO.h"

namespace constellation {
    namespace intrinsics {
        using mlir::edsc::intrinsics::ValueBuilder;
        using mlir::edsc::intrinsics::OperationBuilder;
        using lattice = ValueBuilder<lattice::LatticeOp>;
        using read = ValueBuilder<IO::ReadOp>;
        using write = OperationBuilder<IO::WriteOp>;
    }
}

#endif //LLVM_CONSTELLATION_INTRINSICS_H
