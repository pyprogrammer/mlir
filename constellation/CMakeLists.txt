add_compile_options(-fexceptions)

include_directories(include/)

# Suppress warnings
include_directories(SYSTEM include/constellation/third_party/)

add_subdirectory(lib)

set(LLVM_LINK_COMPONENTS
        Core
        Support
        )

