cmake_minimum_required(VERSION 3.23)
project(FaceRecognizer)
set(CMAKE_CXX_STANDARD 23)

#set(ENV{OPENCV_DIR} "D:\\opencv\\build")
#
#find_package(OpenCv)
#include_directories(${OpenCV_INCLUDE_DIRS})
#
#add_executable(FaceRecognizer main.cpp)
#
#
#target_link_libraries(FaceRecognizer ${OpenCV_LIBS})

set(OpenCV_FOUND "D:\\opencv\\build")

find_package(OpenCV REQUIRED)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FaceRecognizer main.cpp)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(FaceRecognizer ${OpenCV_LIBS})

