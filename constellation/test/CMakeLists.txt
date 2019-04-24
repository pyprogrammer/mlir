macro(add_constellation_example name)
    add_llvm_example(
            constellation-${name} ${ARGN}
    )
    target_link_libraries(
            constellation-${name}
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
endmacro()

set(LLVM_OPTIONAL_SOURCES
        Lattice1.cpp IO1.cpp)

add_constellation_example(Lattice1 Lattice1.cpp)
add_constellation_example(IO1 IO1.cpp)
