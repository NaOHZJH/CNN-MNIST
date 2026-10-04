// gui/predictwindow.cpp

#include "predictwindow.h"
#include "digitwidget.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QRandomGenerator>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <exception>
#include <utility>
#include <vector>

#include "cnn/conv2d.h"
#include "cnn/relu.h"
#include "cnn/pooling.h"
#include "cnn/flatten.h"
#include "cnn/dense.h"
#include "cnn/dataset.h"
#include "cnn/model_io.h"

namespace {

constexpr size_t kImgSide   = 28;
constexpr size_t kImgPixels = kImgSide * kImgSide;  // 784

// 与训练时完全一致的网络结构（方案 A 约定：加载端结构必须与训练端一致）。
// 结构需与 gui/trainer.cpp / src/mnist_test.cpp 中构建的网络保持一致。
cnn::Network buildNetwork() {
    cnn::Network net;
    net.add(cnn::Conv2D(1, 8, 3, 1, 1));    // [N,1,28,28] -> [N,8,28,28]
    net.add(cnn::ReLU());
    net.add(cnn::MaxPool2D(2, 2));          // -> [N,8,14,14]
    net.add(cnn::Conv2D(8, 16, 3, 1, 1));   // -> [N,16,14,14]
    net.add(cnn::ReLU());
    net.add(cnn::MaxPool2D(2, 2));          // -> [N,16,7,7]
    net.add(cnn::Flatten());                // -> [N,784]
    net.add(cnn::Dense(16 * 7 * 7, 10));    // -> [N,10]
    return net;
}

// 对单个样本做预测，返回 (类别, 置信度)。网络输出 logits，置信度取 softmax 后的最大值。
std::pair<int, float> predict(cnn::Network& net, const std::vector<float>& img) {
    cnn::Tensor x({1, 1, kImgSide, kImgSide});
    for (size_t i = 0; i < kImgPixels; ++i) x[i] = img[i];

    cnn::Tensor out  = net.forward(x);              // [1,10] logits
    cnn::Tensor prob = cnn::Tensor::softmax(out);   // [1,10]

    int best = 0;
    float bestv = prob[0];
    for (int c = 1; c < 10; ++c)
        if (prob[c] > bestv) { bestv = prob[c]; best = c; }
    return {best, bestv};
}

} // namespace

PredictWindow::PredictWindow(QWidget* parent) : QWidget(parent) {
    setWindowTitle(QStringLiteral("模型预测"));
    resize(360, 540);

    auto* root = new QVBoxLayout(this);

    // 模型路径
    auto* modelRow = new QHBoxLayout;
    modelRow->addWidget(new QLabel(QStringLiteral("模型:")));
    modelPathEdit_ = new QLineEdit(QStringLiteral("model.bin"));
    modelRow->addWidget(modelPathEdit_, 1);
    loadModelBtn_ = new QPushButton(QStringLiteral("加载模型"));
    modelRow->addWidget(loadModelBtn_);
    root->addLayout(modelRow);

    // 数据路径
    auto* dataRow = new QHBoxLayout;
    dataRow->addWidget(new QLabel(QStringLiteral("数据:")));
    dataPathEdit_ = new QLineEdit(QStringLiteral("data/test.csv"));
    dataRow->addWidget(dataPathEdit_, 1);
    loadDataBtn_ = new QPushButton(QStringLiteral("加载数据"));
    dataRow->addWidget(loadDataBtn_);
    root->addLayout(dataRow);

    // 图片 + 预测
    digit_ = new DigitWidget;
    digit_->setLabels(-1, -1);
    root->addWidget(digit_, 1);

    // 导航
    auto* navRow = new QHBoxLayout;
    prevBtn_ = new QPushButton(QStringLiteral("上一张"));
    nextBtn_ = new QPushButton(QStringLiteral("下一张"));
    randomBtn_ = new QPushButton(QStringLiteral("随机"));
    navRow->addWidget(prevBtn_);
    navRow->addWidget(nextBtn_);
    navRow->addWidget(randomBtn_);
    indexSpin_ = new QSpinBox;
    indexSpin_->setRange(1, 1);
    indexSpin_->setPrefix(QStringLiteral("样本 "));
    navRow->addWidget(indexSpin_, 1);
    root->addLayout(navRow);

    statusLabel_ = new QLabel(QStringLiteral("请先加载模型与测试数据。"));
    statusLabel_->setWordWrap(true);
    root->addWidget(statusLabel_);

    connect(loadModelBtn_, &QPushButton::clicked, this, &PredictWindow::loadModel);
    connect(loadDataBtn_, &QPushButton::clicked, this, &PredictWindow::loadData);
    connect(prevBtn_, &QPushButton::clicked, this, &PredictWindow::prev);
    connect(nextBtn_, &QPushButton::clicked, this, &PredictWindow::next);
    connect(randomBtn_, &QPushButton::clicked, this, &PredictWindow::randomSample);
    connect(indexSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, [this](int oneBased) { showSample(oneBased - 1); });

    updateUiState();
}

void PredictWindow::loadModel() {
    const QString path = modelPathEdit_->text().trimmed();
    if (path.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("加载模型"), QStringLiteral("模型路径不能为空。"));
        return;
    }
    try {
        net_ = buildNetwork();
        cnn::load_model(net_, path.toStdString());
        modelLoaded_ = true;
        statusLabel_->setText(QStringLiteral("模型已加载：%1").arg(path));
    } catch (const std::exception& e) {
        modelLoaded_ = false;
        QMessageBox::warning(this, QStringLiteral("加载模型失败"), QString::fromUtf8(e.what()));
        statusLabel_->setText(QStringLiteral("模型加载失败。"));
    }
    updateUiState();
    if (modelLoaded_ && dataLoaded_) showSample(curIdx_);
}

void PredictWindow::loadData() {
    const QString path = dataPathEdit_->text().trimmed();
    if (path.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("加载数据"), QStringLiteral("数据路径不能为空。"));
        return;
    }
    try {
        cnn::Dataset ds = cnn::load_mnist_csv(path.toStdString(), false, 0);
        if (ds.num_samples() == 0) throw std::runtime_error("数据文件为空");
        images_ = std::move(ds.images);
        dataLoaded_ = true;
        curIdx_ = 0;
        indexSpin_->setRange(1, int(images_.size()));
        statusLabel_->setText(QStringLiteral("已加载 %1 张测试图片").arg(images_.size()));
    } catch (const std::exception& e) {
        dataLoaded_ = false;
        QMessageBox::warning(this, QStringLiteral("加载数据失败"), QString::fromUtf8(e.what()));
        statusLabel_->setText(QStringLiteral("数据加载失败。"));
    }
    updateUiState();
    if (dataLoaded_) showSample(curIdx_);
}

void PredictWindow::prev() { showSample(curIdx_ - 1); }
void PredictWindow::next() { showSample(curIdx_ + 1); }

void PredictWindow::randomSample() {
    if (!dataLoaded_ || images_.empty()) return;
    showSample(int(QRandomGenerator::global()->bounded(quint32(images_.size()))));
}

void PredictWindow::showSample(int idx) {
    if (!dataLoaded_ || images_.empty()) return;
    if (idx < 0) idx = 0;
    if (idx >= int(images_.size())) idx = int(images_.size()) - 1;
    curIdx_ = idx;

    // 显示图片
    QVector<float> px;
    px.reserve(int(kImgPixels));
    for (float v : images_[size_t(curIdx_)]) px.append(v);
    digit_->setImage(px);

    // 预测（测试集无标签，真实类别恒为 -1）
    int pred = -1;
    float conf = 0.0f;
    if (modelLoaded_) {
        const auto [p, c] = predict(net_, images_[size_t(curIdx_)]);
        pred = p;
        conf = c;
    }
    digit_->setLabels(pred, -1);

    // 同步序号框（屏蔽信号，避免递归触发 valueChanged）
    indexSpin_->blockSignals(true);
    indexSpin_->setValue(curIdx_ + 1);
    indexSpin_->blockSignals(false);

    if (modelLoaded_)
        statusLabel_->setText(QStringLiteral("样本 %1 / %2    预测 = %3（置信度 %4%）")
                                  .arg(curIdx_ + 1).arg(int(images_.size()))
                                  .arg(pred).arg(conf * 100.0, 0, 'f', 1));
    else
        statusLabel_->setText(QStringLiteral("样本 %1 / %2    （未加载模型，仅显示图片）")
                                  .arg(curIdx_ + 1).arg(int(images_.size())));
}

void PredictWindow::updateUiState() {
    const bool hasData = dataLoaded_ && !images_.empty();
    prevBtn_->setEnabled(hasData);
    nextBtn_->setEnabled(hasData);
    randomBtn_->setEnabled(hasData);
    indexSpin_->setEnabled(hasData);
}
