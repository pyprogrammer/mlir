//
// Created by Nathan Zhang on 4/17/19.
//

#ifndef CONSTELLATION_INTRINSICS_H
#define CONSTELLATION_INTRINSICS_H

#include "mlir/EDSC/Intrinsics.h"
#include "Lattice.h"
#include "IO.h"

namespace constellation {
    namespace intrinsics {
        using mlir::edsc::intrinsics::ValueBuilder;
        using mlir::edsc::intrinsics::OperationBuilder;
        using lattice = ValueBuilder<lattice::LatticeOp>;
        using read = ValueBuilder<IO::ReadOp>;
        using write = ValueBuilder<IO::WriteOp>;
        using transfer = ValueBuilder<IO::TransferOp>;
    }
}

#endif //CONSTELLATION_INTRINSICS_H
