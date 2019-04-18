add_llvm_example(
        const-input-1
        ConstellationInputExample.cpp
)

target_link_libraries(const-input-1
        PRIVATE
        ConstellationCore
        MLIRAnalysis
        MLIRDialect
        MLIREDSC
        MLIRIR
        MLIRLLVMIR
        MLIRParser
        MLIRPass
        MLIRTransforms
        )
