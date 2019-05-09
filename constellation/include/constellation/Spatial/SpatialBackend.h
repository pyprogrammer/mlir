//
// Created by Nathan Zhang on 4/25/19.
//

#ifndef CONSTELLATION_SPATIALBACKEND_H
#define CONSTELLATION_SPATIALBACKEND_H

#include <string>
#include <set>
#include <vector>

#include "mlir/IR/StandardTypes.h"
#include "mlir/StandardOps/Ops.h"

#include "constellation/core/Backend.h"
#include "constellation/core/Types.h"
#include "constellation/core/ValueIDPass.h"
#include "constellation/core/OpUtils.h"

#include "constellation/core/Lattice.h"

#include "constellation/Spatial/CodeTemplates.h"
#include "constellation/Spatial/SpatialLocationPass.h"

#include "constellation/Spatial/SpatialEmitSupport.h"


namespace constellation::spatial {
    class SpatialModule final: public BackendModule<SpatialModule> {
    public:
        using BackendModule<SpatialModule>::BackendModule;

        void emit(llvm::raw_ostream* os) override;

        static void initPassManager(mlir::PassManager* pm);

    private:

        template<typename T>
        void codegen(std::stringstream& decl, std::stringstream& host, std::stringstream& accel, T) {}

        void genericCodegen(std::stringstream& decl, std::stringstream& host, std::stringstream& accel, mlir::Operation* op);
    };
}

#endif //CONSTELLATION_SPATIALBACKEND_H
