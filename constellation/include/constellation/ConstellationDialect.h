//
// Created by Nathan Zhang on 4/10/19.
//

#include "mlir/IR/Dialect.h"

#include "constellation/ConstCommon.h"

#ifndef LLVM_CONSTELLATION_CONSTELLATIONDIALECT_H
#define LLVM_CONSTELLATION_CONSTELLATIONDIALECT_H

namespace constellation {

    class ConstellationDialect : public mlir::Dialect {

    public:
        explicit ConstellationDialect(mlir::MLIRContext* ctx);

        /// Parse a type registered to this dialect.
        mlir::Type parseType(llvm::StringRef spec, mlir::Location loc) const override;

        /// Print a type registered to this dialect.
        void printType(mlir::Type type, llvm::raw_ostream &os) const override;
    };
}

#endif //LLVM_CONSTELLATION_CONSTELLATIONDIALECT_H
