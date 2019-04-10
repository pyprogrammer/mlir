//
// Created by Nathan Zhang on 4/10/19.
//

#include "mlir/IR/Dialect.h"

#ifndef LLVM_CONSTELLATIONDIALECT_H
#define LLVM_CONSTELLATIONDIALECT_H

namespace constellation {

    class ConstellationDialect : public mlir::Dialect {
    public:
        explicit ConstellationDialect(mlir::MLIRContext* ctx) : mlir::Dialect("constellation", ctx) {};
    };

}

#endif //LLVM_CONSTELLATIONDIALECT_H
