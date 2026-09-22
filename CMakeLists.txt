cmake_minimum_required(VERSION 3.16)

project(cpp_ai_camera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(ai-camera
        src/main.cpp
)

target_link_libraries(ai-camera PRIVATE ${OpenCV_LIBS})