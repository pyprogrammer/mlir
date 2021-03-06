set(LLVM_OPTIONAL_SOURCES
  FileUtilities.cpp
  JitRunner.cpp
  MlirOptMain.cpp
  StorageUniquer.cpp
  TranslateClParser.cpp
)

add_llvm_library(MLIRSupport
  FileUtilities.cpp
  StorageUniquer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support
  )
target_link_libraries(MLIRSupport LLVMSupport)

add_llvm_library(MLIROptMain
  MlirOptMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support
  )
target_link_libraries(MLIROptMain LLVMSupport MLIRPass)

add_llvm_library(MLIRTranslateClParser
  TranslateClParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support
  )
target_link_libraries(MLIRTranslateClParser LLVMSupport)

add_llvm_library(MLIRJitRunner
  JitRunner.cpp
)
target_link_libraries(MLIRJitRunner PRIVATE
  MLIRExecutionEngine
  MLIRIR
  MLIRParser
  MLIRStandardOps
  MLIRTargetLLVMIR
  MLIRTransforms
  MLIRStandardToLLVM
  MLIRSupport
  LLVMCore
  LLVMSupport
)
