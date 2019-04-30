//
// Created by Nathan Zhang on 4/24/19.
//

#ifndef CONSTELLATION_BACKEND_H
#define CONSTELLATION_BACKEND_H

#include <regex>
#include <string>
#include <vector>

#include <inja.hpp>
#include <nlohmann/json.hpp>

#include "mlir/IR/Module.h"
#include "mlir/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

namespace constellation {

// These are meant to be cheap, throwaway objects, useful for parallel codegen of different parameters.

    class BackendFunc {
    public:
        explicit BackendFunc(mlir::Function *func) : func_(func) {}

        virtual ~BackendFunc() = default;

        virtual void emit(llvm::raw_ostream *ostream) {
            llvm_unreachable("Emit not defined on BackendFunc.");
        }

    protected:
        mlir::Function *func_;
    };
}

#endif //CONSTELLATION_BACKEND_H
