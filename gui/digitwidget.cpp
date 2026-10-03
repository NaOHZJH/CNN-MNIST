// gui/digitwidget.cpp

#include "digitwidget.h"

#include <QPainter>

DigitWidget::DigitWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(140, 210);
    setMaximumWidth(190);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
}

void DigitWidget::setImage(const QVector<float>& px) {
    pixels_ = px;
    update();
}

void DigitWidget::setLabels(int pred, int truth) {
    predicted_ = pred;
    truth_ = truth;
    update();
}

void DigitWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);
    p.fillRect(rect(), Qt::white);

    const int cell = 4;      // 每像素放大倍数
    const int side = 28 * cell;  // 112
    const int x0 = (width() - side) / 2;
    const int y0 = 20;

    // 边框
    p.fillRect(x0 - 2, y0 - 2, side + 4, side + 4, QColor(110, 110, 110));

    // 逐像素绘制（值越大越接近白色，MNIST 背景 0 → 黑）
    if (!pixels_.isEmpty()) {
        for (int r = 0; r < 28; ++r) {
            for (int c = 0; c < 28; ++c) {
                int v = 255 - int(pixels_.value(r * 28 + c, 0.0f) * 255.0f + 0.5f);
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                p.fillRect(x0 + c * cell, y0 + r * cell, cell, cell, QColor(v, v, v));
            }
        }
    }

    // 标签文字
    QFont f = p.font();
    f.setPointSize(11);
    f.setBold(true);
    p.setFont(f);
    p.setPen(QColor(30, 30, 30));
    QString s = QStringLiteral("预测: %1\n真实: %2")
                    .arg(predicted_ < 0 ? QStringLiteral("?") : QString::number(predicted_))
                    .arg(truth_ < 0 ? QStringLiteral("?") : QString::number(truth_));
    p.drawText(QRect(0, y0 + side + 8, width(), 42), Qt::AlignHCenter | Qt::AlignTop, s);
}
