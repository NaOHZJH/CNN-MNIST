find_package(Qt6 REQUIRED CONFIG COMPONENTS Widgets)

option(CNN_DEPLOY_QT "Deploy Qt runtime dependencies after building on Windows" ON)

# 同时收集直属文件与各级分组目录中的源码、Qt 头文件。
file(GLOB_RECURSE GUI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE GUI_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
qt_add_executable(mnist_gui
    ${GUI_SOURCES}
    ${GUI_HEADERS})

target_include_directories(mnist_gui PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mnist_gui PRIVATE cnn Qt6::Widgets)
set_target_properties(mnist_gui PROPERTIES
    AUTOMOC ON
    WIN32_EXECUTABLE ON
    # 保持现有运行、调试配置使用的可执行文件输出位置。
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# 使用已找到的 Qt 自带工具部署 DLL、平台插件及匹配的编译器运行库。
# 不强制 --debug：Debug 应用也可以链接这套 Qt 提供的 Release 库。
if(WIN32 AND CNN_DEPLOY_QT)
    get_filename_component(CNN_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    add_custom_command(TARGET mnist_gui POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E env
            "PATH=${CNN_COMPILER_BIN_DIR};$<TARGET_FILE_DIR:Qt6::qmake>;$ENV{PATH}"
            "$<TARGET_FILE:Qt6::windeployqt>"
            --qtpaths "$<TARGET_FILE:Qt6::qtpaths>"
            --no-translations --compiler-runtime
            "$<TARGET_FILE:mnist_gui>"
        COMMENT "Deploying Qt runtime dependencies for mnist_gui"
        VERBATIM)
endif()
