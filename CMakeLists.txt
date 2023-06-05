cmake_minimum_required(VERSION 3.23)
project(FaceRecognizer)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FaceRecognizer main.cpp visualize.h visualize.cpp visualize.cpp)

target_link_libraries(FaceRecognizer ${OpenCV_LIBS})
