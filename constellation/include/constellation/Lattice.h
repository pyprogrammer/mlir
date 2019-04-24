//
// Created by Nathan Zhang on 4/17/19.
//

#ifndef LLVM_CONSTELLATION_SIMPLEXLATTICE_H
#define LLVM_CONSTELLATION_SIMPLEXLATTICE_H

#include <string>
#include <map>

#include "constellation/ConstCommon.h"
#include "constellation/OpCommon.h"

namespace constellation {
    namespace lattice {

        enum class LatticeType {
            SIMPLEX,
            HYPERCUBE,

            ENUM_LAST = HYPERCUBE
        };

        class LatticeOp : public mlir::Op<LatticeOp, mlir::OpTrait::NOperands<2>::Impl,
                mlir::OpTrait::OneResult,
                mlir::OpTrait::HasNoSideEffect,
                mlir::OpTrait::ResultsAreFloatLike> {
        public:

            using Op::Op;

            //////////////////////////////////////////////////////////////////////////////
            // Hooks to customize the behavior of this op.
            //////////////////////////////////////////////////////////////////////////////
            static llvm::StringRef getOperationName() { return "constellation.lattice"; }

            // Simplex: Params: sizes: List[dim_sizes], Params: List[len(sizes)]
            static void build(mlir::Builder *b, mlir::OperationState *state,
                              mlir::Value *input, mlir::Value *params, LatticeType latticeType);

            mlir::LogicalResult verify();

            //////////////////////////////////////////////////////////////////////////////
            // Op-specific functionality.
            //////////////////////////////////////////////////////////////////////////////
            unsigned ndim();

            mlir::Value *input();

            mlir::Value *params();
        };
    }
}

#endif //LLVM_CONSTELLATION_SIMPLEXLATTICE_H
