// gui/main.cpp
// Qt6 可视化程序入口。
// 运行方式（在项目根目录下）：./build/mnist_gui.exe
#include <QApplication>

#include "mainwindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}
