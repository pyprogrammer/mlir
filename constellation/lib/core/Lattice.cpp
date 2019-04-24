//
// Created by Nathan Zhang on 4/17/19.
//

#include "constellation/core/Lattice.h"

namespace constellation {
    namespace lattice {

        void LatticeOp::build(mlir::Builder *b, mlir::OperationState *state, mlir::Value *input, mlir::Value *params,
                              LatticeType latticeType) {
            assert(input->getType().isa<mlir::VectorType>() && "Input to a Lattice should be a vector");
            assert(params->getType().isa<mlir::RankedTensorType>() && "Lattice parameters should be a ranked tensor.");

            state->addOperands({input, params});
            state->addTypes({params->getType().cast<mlir::RankedTensorType>().getElementType()});
            state->addAttribute("type", toIntegerAttr(b, latticeType));
        }

        unsigned LatticeOp::ndim() {
            return getOperand(0)->getType().cast<mlir::VectorType>().getNumElements();
        }

        mlir::LogicalResult LatticeOp::verify() {
            unsigned param_dim = getOperand(1)->getType().cast<mlir::RankedTensorType>().getRank();
            // check if all dimensions agree
            if (ndim() != param_dim) {
                return emitOpError("Lattice input and parameter dimensions should agree.");
            }
            return mlir::success();
        }

        mlir::Value *LatticeOp::input() {
            return getOperand(0);
        }

        mlir::Value *LatticeOp::params() {
            return getOperand(1);
        }
    }
}
