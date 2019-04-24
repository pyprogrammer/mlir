//
// Created by Nathan Zhang on 4/17/19.
//

#ifndef CONSTELLATION_INTRINSICS_H
#define CONSTELLATION_INTRINSICS_H

#include "mlir/EDSC/Intrinsics.h"
#include "constellation/Lattice.h"
#include "constellation/IO.h"

namespace constellation {
    namespace intrinsics {
        using mlir::edsc::intrinsics::ValueBuilder;
        using mlir::edsc::intrinsics::OperationBuilder;
        using lattice = ValueBuilder<lattice::LatticeOp>;
        using read = ValueBuilder<IO::ReadOp>;
        using write = ValueBuilder<IO::WriteOp>;
    }
}

#endif //CONSTELLATION_INTRINSICS_H
