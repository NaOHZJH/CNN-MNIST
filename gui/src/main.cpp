// gui/src/main.cpp
// Qt6 可视化程序入口。
// 运行方式（在项目根目录下）：./build/qt-mingw-release/mnist_gui.exe
#include <QApplication>

#include "windows/mainwindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}
