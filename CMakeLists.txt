cmake_minimum_required(VERSION 3.21)
project(cnn_mnist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- VS Code 调试读取这些缓存项；本机路径由 CMakeUserPresets.json 提供。
set(CNN_GDB_PATH "" CACHE FILEPATH "GDB executable used by VS Code")
set(CNN_RUNTIME_DIRECTORIES "" CACHE STRING "Runtime DLL directories used by VS Code")
mark_as_advanced(CNN_GDB_PATH CNN_RUNTIME_DIRECTORIES)

# ---- CNN 核心库：递归收集实现文件，排除应用入口 ----
file(GLOB_RECURSE CNN_LIB_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM CNN_LIB_SOURCES
     "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"       # cnn_demo 的 main
     "${CMAKE_CURRENT_SOURCE_DIR}/src/mnist_test.cpp" # mnist_test 的 main
)
add_library(cnn STATIC ${CNN_LIB_SOURCES})
target_include_directories(cnn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---- 1) 前向传播演示（旧示例，可选）----
add_executable(cnn_demo src/main.cpp)
target_link_libraries(cnn_demo PRIVATE cnn)

# ---- 2) 控制台训练/测试程序 ----
add_executable(mnist_test src/mnist_test.cpp)
target_link_libraries(mnist_test PRIVATE cnn)

# ---- 3) Qt6 可视化程序（可用 -DCNN_BUILD_GUI=OFF 关闭）----
# 本机 Qt / MinGW / Ninja 路径统一由 CMakeUserPresets.json 配置。
option(CNN_BUILD_GUI "Build the Qt6 Widgets application" ON)

if(CNN_BUILD_GUI)
    add_subdirectory(gui)
endif()
