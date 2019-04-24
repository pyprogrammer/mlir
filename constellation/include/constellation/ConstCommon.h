//
// Created by Nathan Zhang on 4/17/19.
//

#ifndef CONSTELLATION_COMMON_H
#define CONSTELLATION_COMMON_H

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
#include "mlir/IR/OpImplementation.h"

/// A basic function builder
inline mlir::Function *makeFunction(mlir::Module &module, llvm::StringRef name,
                                    llvm::ArrayRef<mlir::Type> types,
                                    llvm::ArrayRef<mlir::Type> resultTypes) {
    auto *context = module.getContext();
    auto *function = new mlir::Function(
            mlir::UnknownLoc::get(context), name,
            mlir::FunctionType::get({types}, resultTypes, context));
    function->addEntryBlock();
    module.getFunctions().push_back(function);
    return function;
}

/// A basic pass manager pre-populated with cleanup passes.
inline std::unique_ptr<mlir::PassManager> cleanupPassManager() {
    std::unique_ptr<mlir::PassManager> pm(new mlir::PassManager());
    pm->addPass(mlir::createCanonicalizerPass());
    pm->addPass(mlir::createSimplifyAffineStructuresPass());
    pm->addPass(mlir::createCSEPass());
    pm->addPass(mlir::createCanonicalizerPass());
    return pm;
}

inline void cleanupAndPrintFunction(mlir::Function *f) {
    bool printToOuts = true;
    auto check = [f, &printToOuts](mlir::LogicalResult result) {
        if (failed(result)) {
            f->getContext()->emitError(f->getLoc(),
                                       "Verification and cleanup passes failed");
            printToOuts = false;
        }
    };
    auto pm = cleanupPassManager();
    check(f->getModule()->verify());
    check(pm->run(f->getModule()));
    if (printToOuts)
        f->print(llvm::outs());
}

#endif //CONSTELLATION_COMMON_H
