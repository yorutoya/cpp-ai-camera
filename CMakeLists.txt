cmake_minimum_required(VERSION 3.16)

project(cpp_ai_camera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn highgui videoio)
if(OpenCV_VERSION VERSION_LESS "4.0")
    message(FATAL_ERROR "OpenCV 4 or newer is required")
endif()

add_executable(ai-camera
        src/main.cpp
        src/YoloDetector.cpp
        src/CameraAnalyzer.cpp
)

target_include_directories(ai-camera PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ai-camera PRIVATE
        ${OpenCV_LIBS}
)
if(MSVC)
    target_compile_options(ai-camera PRIVATE /W4)
else()
    target_compile_options(ai-camera PRIVATE -Wall -Wextra -Wpedantic)
endif()
