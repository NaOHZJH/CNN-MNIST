// gui/digitwidget.h
// 显示一个 28×28 手写数字（灰度），并在下方显示网络的预测类别与真实类别。
#pragma once

#include <QVector>
#include <QWidget>

class DigitWidget : public QWidget {
    Q_OBJECT
public:
    explicit DigitWidget(QWidget* parent = nullptr);

    void setImage(const QVector<float>& pixels);  // 784 个 [0,1] 像素，按行主序
    void setLabels(int predicted, int truth);     // 更新下方文字（-1 表示未知）

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QVector<float> pixels_;  // 784 个 [0,1]
    int predicted_ = -1;
    int truth_ = -1;
};
