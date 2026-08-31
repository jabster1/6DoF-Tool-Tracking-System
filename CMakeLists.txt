cmake_minimum_required(VERSION 3.20)
project(ToolTrackerBenchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OpenCV ─────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED
    COMPONENTS core imgproc imgcodecs highgui videoio
)
message(STATUS "OpenCV version : ${OpenCV_VERSION}")

# ── CUDA / TensorRT (Jetson JetPack paths) ──────────────────────────────────
find_package(CUDAToolkit REQUIRED)

set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu)
set(TENSORRT_LIB_DIR /usr/lib/aarch64-linux-gnu)

# ── tool_tracker_trt ─────────────────────────────────────────────────────
add_executable(tool_tracker_trt tool_tracker_onnx.cpp)

target_include_directories(tool_tracker_trt PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(tool_tracker_trt PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TENSORRT_LIB_DIR}/libnvinfer.so
    ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
    ${TENSORRT_LIB_DIR}/libnvonnxparser.so
)

install(TARGETS tool_tracker_trt
        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/build/bin)