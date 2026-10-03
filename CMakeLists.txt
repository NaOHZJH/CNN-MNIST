cmake_minimum_required(VERSION 3.16)
project(cnn_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 收集 src/ 下所有 .cpp（含 main.cpp）。新增源文件后重新构建即可自动纳入。
file(GLOB CNN_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(cnn_demo ${CNN_SOURCES})
target_include_directories(cnn_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
