cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(project_irgpua LANGUAGES CXX CUDA)

set(RAFT_NVTX ON)
include(fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(tp_reduce)

rapids_cpm_init()
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)

find_package(OpenMP REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
set(CUDA_ARCHITECTURES "75")
set(CMAKE_CUDA_ARCHITECTURES "75")

include_directories(src)

add_executable(main
            src/main.cu
            src/main_kernel.cu
            src/main_cpu.cu
            src/main_indus.cu

            src/kernel/fix_gpu.cu
            src/kernel/find_first_value.cu
            src/kernel/scan.cu
            src/kernel/map.cu
            src/kernel/histo.cu
            src/kernel/calculate_mask.cu
            src/kernel/scattering.cu
                        
            src/cpu/fix_cpu.cu

            src/indus/fix_gpu.cu
)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)
