//
// Created by Nathan Zhang on 2019-05-08.
//

#ifndef CONSTELLATION_VALUENUMBERINGPASS_H
#define CONSTELLATION_VALUENUMBERINGPASS_H

#include <string>

#include "mlir/Pass/Pass.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

namespace constellation::core::passes {
    class ValueIDPass : public mlir::FunctionPass<ValueIDPass> {
        void runOnFunction() override;
        static std::string kAttributeName;

    public:
        static int getID(mlir::Operation* op);
    };
}

#endif //CONSTELLATION_VALUENUMBERINGPASS_H
