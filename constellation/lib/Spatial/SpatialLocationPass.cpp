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
        Memory defaultMem(Memory::Location::CPU, deviceIndex);

        auto& func = getFunction();

        func.walk([defaultMem, this](mlir::Operation* op){
            if (auto transfer = op->dyn_cast<IO::TransferOp>()) {
                Memory previous = defaultMem;
                if (auto parentOp = transfer.getOperand()->getDefiningOp()) {
                    previous = getMemory(parentOp);
                }
                if (transfer.getInputMem() != previous) {
                    signalPassFailure();
                }
                setMemory(op, transfer.getOutputMem());
            } else {
                Memory previous = defaultMem;
                if (op->getNumOperands() != 0) {
                    previous = getMemory(op->getOperand(0)->getDefiningOp());
                }
                setMemory(op, previous);
            }
        });
    }

    Memory SpatialLocationPass::getMemory(mlir::Operation *op) {
        return Memory(op->getAttrOfType<mlir::IntegerAttr>(kAttributeName).getInt());
    }

    void SpatialLocationPass::setMemory(mlir::Operation *op, Memory mem) {
        op->setAttr(kAttributeName,
                    mlir::IntegerAttr::get(
                            mlir::IntegerType::get(sizeof(Memory::MLIRMemorySpaceType) * 8,
                                                   &getContext()), Memory::MLIRMemorySpaceType(mem)));
    }

    static mlir::PassRegistration<SpatialLocationPass> pass("spatial-loc", "Spatial Location");
}
