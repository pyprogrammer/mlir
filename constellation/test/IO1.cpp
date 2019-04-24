//===- Example.cpp - Our running example ----------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// RUN: %p/test | FileCheck %s

#include <constellation/core/ConstellationDialect.h>
#include "TestHarness.h"
#include "mlir/IR/Function.h"

#include "constellation/core/ConstCommon.h"
#include "constellation/core/Intrinsics.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

TEST_FUNC(simplex_with_io) {
    MLIRContext context;
    Module module(&context);

    auto fltType = mlir::FloatType::getF64(&context);
    auto vecType = mlir::VectorType::get({5}, fltType);
    auto paramType = mlir::RankedTensorType::get({2, 4, 8, 16, 32}, fltType);
    {
        Function *f = makeFunction(module, "load_lattice", {vecType}, {});
        ScopedContext sc(f);
        ValueHandle in(f->getArgument(0));
        std::string path = "/dev/null";
        auto params = constellation::intrinsics::read({path, constellation::IO::AccessMode::FULL, paramType});
        auto lat = constellation::intrinsics::lattice({in, params, constellation::lattice::LatticeType::SIMPLEX});
        (void) constellation::intrinsics::write({path, constellation::IO::AccessMode::STREAM, lat.getValue()});
        ret();
        cleanupAndPrintFunction(f);
    }
}

int main() {
    mlir::registerDialect<constellation::ConstellationDialect>();
    RUN_TESTS();
    return 0;
}
