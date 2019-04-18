//
// Created by Nathan Zhang on 4/17/19.
//

#ifndef LLVM_CONSTELLATION_INTRINSICS_H
#define LLVM_CONSTELLATION_INTRINSICS_H

#include "mlir/EDSC/Intrinsics.h"

namespace constellation {
namespace intrinsics {

using range = mlir::edsc::intrinsics::ValueBuilder<RangeOp>;
using slice = mlir::edsc::intrinsics::ValueBuilder<SliceOp>;
using view = mlir::edsc::intrinsics::ValueBuilder<ViewOp>;

}
}

#endif //LLVM_CONSTELLATION_INTRINSICS_H
