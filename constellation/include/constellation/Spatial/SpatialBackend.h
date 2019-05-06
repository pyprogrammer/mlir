//
// Created by Nathan Zhang on 4/25/19.
//

#ifndef CONSTELLATION_SPATIALBACKEND_H
#define CONSTELLATION_SPATIALBACKEND_H

#include <string>
#include <set>

#include "mlir/IR/StandardTypes.h"

#include "constellation/core/Backend.h"
#include "constellation/core/Types.h"
#include "constellation/Spatial/CodeTemplates.h"
#include "constellation/Spatial/SpatialLocationPass.h"


namespace constellation::spatial {
    class SpatialFunc final: public BackendFunc<SpatialFunc> {
    public:
        using BackendFunc<SpatialFunc>::BackendFunc;

        void emit(llvm::raw_ostream* os) override;

        static void initPassManager(mlir::PassManager* pm);
    };

}

#endif //CONSTELLATION_SPATIALBACKEND_H
