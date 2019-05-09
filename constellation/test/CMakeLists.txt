macro(add_constellation_example name)
    add_llvm_example(
            constellation-${name} ${ARGN}
    )
    target_link_libraries(
            constellation-${name}
            PRIVATE
            ConstellationCore
            ConstellationSpatial
            MLIRAnalysis
            MLIRDialect
            MLIREDSC
            MLIRIR
            MLIRLLVMIR
            MLIRParser
            MLIRPass
            MLIRTransforms
            MLIRStandardOps
    )
endmacro()

add_subdirectory(core/)
add_subdirectory(Spatial/)
