cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)

# Set the project name
project(fused_attn LANGUAGES C CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# Simple GPU capabilitiy test util
set(file ${PROJECT_BINARY_DIR}/detect_cuda_arch.cu)
file(WRITE ${file} ""
    "#include <cuda_runtime.h>\n"
    "#include <cstdio>\n"
    "int main() {\n"
    "   int count = 0;\n"
    "   if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
    "   if (count == 0) return -1;\n"
    "   cudaDeviceProp prop;\n"
    "   if (cudaSuccess != cudaGetDeviceProperties(&prop, 0)) {\n"
    "       return -1;\n"
    "   }\n"
    "   std::printf(\"%d%d\", prop.major, prop.minor);\n"
    "   return 0;\n"
    "}\n"
)

# Define CUDA capabilities
try_run(
    run_result
    compile_result
    ${PROJECT_BINARY_DIR}
    ${file}
    RUN_OUTPUT_VARIABLE compute_capabilities
)

message(STATUS "Found architecture: ${compute_capabilities}")

# Set the CUDA architecture to compile for
set(CUDA_ARCH compute_capabilities)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=compute_${compute_capabilities} -code=sm_${compute_capabilities} -g -G")

# Add the CUDA include directories
include_directories(include ${CUDA_INCLUDE_DIRS})

# Add the CUDA source files
add_executable(
    test
    src/test.cu
)

# Set the CUDA architecture for the target
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})

# Link against CUDA libraries
target_link_libraries(test ${CUDA_LIBRARIES})