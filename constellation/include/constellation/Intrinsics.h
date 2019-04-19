//
// Created by Nathan Zhang on 4/17/19.
//

#ifndef LLVM_CONSTELLATION_INTRINSICS_H
#define LLVM_CONSTELLATION_INTRINSICS_H

#include "mlir/EDSC/Intrinsics.h"
#include "constellation/Lattice.h"

namespace constellation {
namespace intrinsics {

using lattice = mlir::edsc::intrinsics::ValueBuilder<LatticeOp>;

}
}

#endif //LLVM_CONSTELLATION_INTRINSICS_H
