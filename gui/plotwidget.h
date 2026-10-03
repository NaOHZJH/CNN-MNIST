// gui/plotwidget.h
// 简单的折线图控件：横轴为样本序号（0,1,2,...），纵轴自动缩放。
// 用 QPainter 手绘，避免依赖 Qt Charts 模块。
#pragma once

#include <QVector>
#include <QWidget>

class PlotWidget : public QWidget {
    Q_OBJECT
public:
    explicit PlotWidget(QWidget* parent = nullptr);

    void setTitle(const QString& title);
    void setYLabel(const QString& label);
    void setXLabel(const QString& label);

    void clear();          // 清空曲线
    void addPoint(double y);  // 追加一个数据点（x = 当前点数）

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QString title_;
    QString ylabel_;
    QString xlabel_;
    QVector<double> ys_;
};
