// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @access_chain(%arg0 : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>,
                     %arg1 : i32, %arg2 : i32) {
    // CHECK: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
    // CHECK-NEXT: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
    %1 = spv.AccessChain %arg0[%arg1] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
    %2 = spv.AccessChain %arg0[%arg1, %arg2] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
    spv.Return
  }
}