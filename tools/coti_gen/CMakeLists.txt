add_llvm_library(coti_gen MODULE coti_gen.cpp PLUGIN_TOOL clang)
target_include_directories(coti_gen PRIVATE ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
