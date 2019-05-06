//
// Created by Nathan Zhang on 2019-05-01.
//

#include "constellation/Spatial/SpatialLocationPass.h"

namespace constellation::spatial::passes {

    namespace {
        std::set<std::string> accelOps = {
            lattice::LatticeOp::getOperationName()
        };

        bool isAccelable(mlir::Operation* op) {
            return accelOps.find(op->getName().getStringRef().str()) != accelOps.end();
        }
    }

    std::string SpatialLocationPass::kAttributeName = "SpatialLocationPass";

    void SpatialLocationPass::runOnFunction() {
        // TODO: currently assumes that all spatial funcs are pinned to FPGA-0 and such.
        // Unless we have a multi-FPGA device, we shouldn't have a problem with this assumption
        int deviceIndex = 0;

        auto& func = getFunction();
        Memory current(Memory::Location::CPU, deviceIndex);

        func.walk([this, &current](mlir::Operation* op){
            if (auto transfer = op->dyn_cast<IO::TransferOp>()) {
                if (transfer.getInputMem() != current) {
                    signalPassFailure();
                }
                current = transfer.getOutputMem();
            } else {
                op->setAttr(kAttributeName,
                        mlir::IntegerAttr::get(
                                mlir::IntegerType::get(sizeof(Memory::MLIRMemorySpaceType) * 8,
                                &getContext()), Memory::MLIRMemorySpaceType(current)));
            }
        });
    }
    static mlir::PassRegistration<SpatialLocationPass> pass("spatial-loc", "Spatial Location");
}
