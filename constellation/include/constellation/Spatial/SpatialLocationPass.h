//
// Created by Nathan Zhang on 2019-05-01.
//

#ifndef CONSTELLATION_SPATIALLOCATIONPASS_H
#define CONSTELLATION_SPATIALLOCATIONPASS_H

#include <set>
#include <string>

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

#include "constellation/core/Intrinsics.h"

namespace constellation::spatial::passes {
class SpatialLocationPass : public mlir::FunctionPass<SpatialLocationPass> {
    void runOnFunction() override;
    void setMemory(mlir::Operation* op, Memory mem);
public:
    static std::string kAttributeName;
    static Memory getMemory(mlir::Operation* op);
};
}

#endif //CONSTELLATION_SPATIALLOCATIONPASS_H
