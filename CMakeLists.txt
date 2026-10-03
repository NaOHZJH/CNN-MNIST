cmake_minimum_required(VERSION 3.16)
project(cnn_mnist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- CNN 核心库（除可执行文件入口外的所有 src/*.cpp）----
file(GLOB CNN_LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
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

# ---- 3) Qt6 可视化程序（可选，需安装 Qt6 Widgets）----
# 构建时请加：-DCMAKE_PREFIX_PATH=C:/Qt/6.12.0/mingw_64
find_package(Qt6 QUIET COMPONENTS Widgets)
if(Qt6_FOUND)
    file(GLOB GUI_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/gui/*.cpp")
    add_executable(mnist_gui ${GUI_SOURCES})
    target_link_libraries(mnist_gui PRIVATE cnn Qt6::Widgets)
    set_target_properties(mnist_gui PROPERTIES AUTOMOC ON)   # Q_OBJECT 需要 moc
else()
    message(STATUS "Qt6 未找到，跳过 mnist_gui（如需构建请加 -DCMAKE_PREFIX_PATH=<Qt路径>）")
endif()
