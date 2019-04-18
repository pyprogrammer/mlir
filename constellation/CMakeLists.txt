include_directories(include/)
add_subdirectory(lib)
add_subdirectory(test/)

set(LLVM_LINK_COMPONENTS
        Core
        Support
        )

add_custom_target(ConstellationCompiler)

add_dependencies(ConstellationCompiler ConstellationCore)
