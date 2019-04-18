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

#include <constellation/ConstellationDialect.h>
#include "TestHarness.h"
#include "mlir/IR/Function.h"

#include "constellation/ConstCommon.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

TEST_FUNC(view_op) {
    MLIRContext context;
    Module module(&context);

    // Let's be lazy and define some custom ops that prevent DCE.
//    CustomOperation<OperationHandle> some_consumer("some_consumer");
//
//    some_consumer();
//    ret();

}

int main() {
    mlir::registerDialect<constellation::ConstellationDialect>();
    RUN_TESTS();
    return 0;
}
