//
// Created by Nathan Zhang on 4/24/19.
//

#ifndef CONSTELLATION_BACKEND_H
#define CONSTELLATION_BACKEND_H

#include <regex>
#include <string>
#include <vector>
#include <memory>

#include <inja.hpp>
#include <nlohmann/json.hpp>

#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

namespace constellation {

// These are meant to be cheap, throwaway objects, useful for parallel codegen of different parameters.

    template<typename Derived>
    class BackendModule {
    public:
        explicit BackendModule(mlir::Module *module) : module_(module) {
            Derived::initPassManager(pm_.get());
            if (mlir::failed(pm_->run(module_))) {

            }
        }

        virtual ~BackendModule() = default;

        virtual void emit(llvm::raw_ostream *ostream) {
            llvm_unreachable("Emit not defined on BackendModule.");
        }

        static void initPassManager(mlir::PassManager*) {}

    protected:

        mlir::Module *module_;
        std::unique_ptr<mlir::PassManager> pm_ = std::make_unique<mlir::PassManager>();
    };
}

#endif //CONSTELLATION_BACKEND_H
