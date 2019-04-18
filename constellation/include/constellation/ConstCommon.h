//
// Created by Nathan Zhang on 4/17/19.
//

#ifndef LLVM_CONSTELLATION_COMMON_H
#define LLVM_CONSTELLATION_COMMON_H

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

// For debugging
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

// For Printing
#include "mlir/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"

#endif //LLVM_CONSTELLATION_COMMON_H
