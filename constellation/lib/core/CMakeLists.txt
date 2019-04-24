add_llvm_library(ConstellationCore
        ConstellationDialect.cpp
        Lattice.cpp
        IO.cpp
        )

target_link_libraries(
        ConstellationCore
        PRIVATE
        MLIRAnalysis
        MLIRDialect
        MLIREDSC
        MLIRIR
        MLIRLLVMIR
        MLIRParser
        MLIRPass
        MLIRTransforms
)
