
cmake_minimum_required(VERSION 3.5)

##
## PROJECT NAME
##
set(CMAKE_CUDA_ARCHITECTURES 75)
project(KINFU LANGUAGES CUDA CXX)
##
## COMPILER SETTINGS
##
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75;-G;-g)


# project(KINFU LANGUAGE CXX CUDA)
##opencv
find_package(OpenCV CONFIG REQUIRED)
set(OPENCV_LIBS opencv_ml opencv_dnn opencv_dpm opencv_hdf opencv_cudawarping opencv_cudaimgproc)

##
file(GLOB KFUSION_SRC
    "kfusion/src/*.cpp"
    "kfusion/src/*.cu"
)

set(KFUSION_INC_DIRS "kfusion/include")

add_executable(KINFU main.cpp ${KFUSION_SRC})

target_include_directories(KINFU PRIVATE ${KFUSION_INC_DIRS})

target_link_libraries(KINFU PRIVATE ${OPENCV_LIBS})

set_target_properties(KINFU 
PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


