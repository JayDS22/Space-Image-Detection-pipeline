cmake_minimum_required(VERSION 3.14)
project(SpaceSensingImgProc
    VERSION 1.0.0
    DESCRIPTION "Image processing pipeline for space surveillance detection"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Reasonable default optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# Find OpenCV - we need core, imgproc, and imgcodecs at minimum
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Library containing the shared detection code
add_library(sslab_imgproc
    src/image_processing.cpp
    src/star_detector.cpp
    src/streak_detector.cpp
)
target_include_directories(sslab_imgproc PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(sslab_imgproc PUBLIC ${OpenCV_LIBS})

# Main executable
add_executable(space_sensing_pipeline src/main.cpp)
target_link_libraries(space_sensing_pipeline PRIVATE sslab_imgproc)

# Test executable
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE sslab_imgproc)

# CTest integration
enable_testing()
add_test(NAME pipeline_tests COMMAND test_pipeline)

# Install targets (optional, for packaging)
install(TARGETS space_sensing_pipeline DESTINATION bin)
install(TARGETS sslab_imgproc DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
