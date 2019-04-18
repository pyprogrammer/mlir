//
// Created by Nathan Zhang on 4/17/19.
//

#ifndef LLVM_CONSTELLATION_SIMPLEXLATTICE_H
#define LLVM_CONSTELLATION_SIMPLEXLATTICE_H

#include <string>
#include <map>

#include "constellation/ConstCommon.h"

namespace constellation {

    class LatticeOp : public mlir::Op<LatticeOp, mlir::OpTrait::NOperands<2>::Impl,
            mlir::OpTrait::OneResult,
            mlir::OpTrait::HasNoSideEffect> {
    public:
        enum LatticeType {
            SIMPLEX,
            HYPERCUBE
        };

        //////////////////////////////////////////////////////////////////////////////
        // Hooks to customize the behavior of this op.
        //////////////////////////////////////////////////////////////////////////////
        static llvm::StringRef getOperationName() { return "constellation.lattice"; }
        // Simplex: Params: sizes: List[dim_sizes], Params: List[len(sizes)]
        static void build(mlir::Builder *b, mlir::OperationState *result,
                          mlir::Value *input, mlir::Value *params, LatticeType latticeType);
        mlir::LogicalResult verify();
        static bool parse(mlir::OpAsmParser *parser, mlir::OperationState *result);
        void print(mlir::OpAsmPrinter *p);

        //////////////////////////////////////////////////////////////////////////////
        // Op-specific functionality.
        //////////////////////////////////////////////////////////////////////////////
        unsigned ndim();
        LatticeType getLatticeType();

    private:
        static std::map<LatticeType, std::string> name_map_;
        static std::string& TypeToString(LatticeType latticeType);
    };
}

#endif //LLVM_CONSTELLATION_SIMPLEXLATTICE_H
