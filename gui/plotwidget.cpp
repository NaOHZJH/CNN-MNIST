// gui/plotwidget.cpp

#include "plotwidget.h"

#include <QPainter>
#include <QPainterPath>

#include <algorithm>
#include <cmath>

PlotWidget::PlotWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(320, 180);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void PlotWidget::setTitle(const QString& t) { title_ = t; update(); }
void PlotWidget::setYLabel(const QString& l) { ylabel_ = l; update(); }
void PlotWidget::setXLabel(const QString& l) { xlabel_ = l; update(); }

void PlotWidget::clear() {
    ys_.clear();
    update();
}

void PlotWidget::addPoint(double y) {
    ys_.append(y);
    update();
}

void PlotWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const int marginLeft = 60, marginRight = 18, marginTop = 26, marginBottom = 32;
    const QRect plot(marginLeft, marginTop, width() - marginLeft - marginRight,
                     height() - marginTop - marginBottom);
    if (plot.width() < 10 || plot.height() < 10) return;

    // 背景
    p.fillRect(rect(), QColor(255, 255, 255));
    p.fillRect(plot, QColor(248, 248, 248));
    p.setPen(QColor(180, 180, 180));
    p.drawRect(plot);

    // 标题
    QFont bold = p.font();
    bold.setBold(true);
    p.setFont(bold);
    p.setPen(QColor(30, 30, 30));
    p.drawText(QRect(0, 2, width(), marginTop - 4), Qt::AlignCenter, title_);
    p.setFont(QFont());

    if (ys_.isEmpty()) {
        p.setPen(QColor(150, 150, 150));
        p.drawText(plot, Qt::AlignCenter, QStringLiteral("（暂无数据）"));
        return;
    }

    // 纵轴范围（自动缩放，留 8% 边距）
    double ymin = ys_[0], ymax = ys_[0];
    for (double v : ys_) {
        ymin = std::min(ymin, v);
        ymax = std::max(ymax, v);
    }
    if (ymax - ymin < 1e-9) {
        double pad = std::max(1.0, std::fabs(ymax) * 0.1);
        ymin -= pad;
        ymax += pad;
    } else {
        double pad = (ymax - ymin) * 0.08;
        ymin -= pad;
        ymax += pad;
    }

    const int n = ys_.size();
    const double xmax = (n > 1) ? double(n - 1) : 1.0;

    // 横向网格 + 纵轴刻度
    const int nyTicks = 4;
    for (int i = 0; i <= nyTicks; ++i) {
        double yv = ymin + (ymax - ymin) * i / nyTicks;
        int y = plot.bottom() - int((yv - ymin) / (ymax - ymin) * plot.height());
        p.setPen(QColor(225, 225, 225));
        p.drawLine(plot.left(), y, plot.right(), y);
        p.setPen(QColor(120, 120, 120));
        p.drawText(QRect(0, y - 8, marginLeft - 6, 16), Qt::AlignRight | Qt::AlignVCenter,
                   QString::number(yv, 'f', 2));
    }

    // 横轴刻度（数据点下标）
    const int xTicks = std::min(6, std::max(1, n));
    p.setPen(QColor(120, 120, 120));
    for (int i = 0; i <= xTicks; ++i) {
        int idx = int(double(n - 1) * i / xTicks);
        int x = plot.left() + int(double(idx) / xmax * plot.width());
        p.drawText(QRect(x - 20, plot.bottom() + 4, 40, 16), Qt::AlignCenter,
                   QString::number(idx));
    }

    // 轴标签
    p.save();
    p.translate(14, plot.center().y());
    p.rotate(-90);
    p.drawText(QRect(-60, -10, 120, 20), Qt::AlignCenter, ylabel_);
    p.restore();
    p.drawText(QRect(plot.left(), plot.bottom() + 18, plot.width(), 14),
               Qt::AlignCenter, xlabel_);

    // 曲线
    QPainterPath path;
    QVector<QPointF> pts;
    for (int i = 0; i < n; ++i) {
        double x = double(i) / xmax;
        double y = (ys_[i] - ymin) / (ymax - ymin);
        QPointF pt(plot.left() + x * plot.width(), plot.bottom() - y * plot.height());
        pts.append(pt);
        if (i == 0) path.moveTo(pt);
        else path.lineTo(pt);
    }
    p.setPen(QPen(QColor(37, 116, 227), 2));
    p.drawPath(path);

    // 数据点
    p.setBrush(QColor(37, 116, 227));
    p.setPen(Qt::NoPen);
    for (const QPointF& pt : pts) p.drawEllipse(pt, 3, 3);
}
