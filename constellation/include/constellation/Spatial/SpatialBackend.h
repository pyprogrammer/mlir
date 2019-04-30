//
// Created by Nathan Zhang on 4/25/19.
//

#ifndef CONSTELLATION_SPATIALBACKEND_H
#define CONSTELLATION_SPATIALBACKEND_H

#include <string>

#include "constellation/core/Backend.h"
#include "mlir/IR/StandardTypes.h"

#include "constellation/Spatial/CodeTemplates.h"

namespace constellation {
    namespace spatial {
        class SpatialFunc final: public BackendFunc {
        public:
            using BackendFunc::BackendFunc;

            void emit(llvm::raw_ostream* os) override;
        private:
            static const std::string topTemplate;
        };
    }
}

#endif //CONSTELLATION_SPATIALBACKEND_H
