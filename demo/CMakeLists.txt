

project(demo LANGUAGES CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)     # 先用86，后面我再帮你调

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-G)

add_executable(demo demo.cu)
