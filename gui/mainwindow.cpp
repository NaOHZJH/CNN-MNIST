// gui/mainwindow.cpp

#include "mainwindow.h"
#include "plotwidget.h"
#include "digitwidget.h"
#include "trainer.h"

#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QThread>
#include <QVBoxLayout>

#include <exception>

#include "cnn/dataset.h"

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("MNIST CNN 训练可视化"));
    resize(1000, 680);
    buildUi();
}

void MainWindow::buildUi() {
    auto* central = new QWidget(this);
    setCentralWidget(central);
    auto* root = new QVBoxLayout(central);

    // ---- 顶部：参数面板 ----
    auto* ctrl = new QGroupBox(QStringLiteral("训练参数"));
    auto* grid = new QGridLayout(ctrl);

    auto addSpin = [&](int row, int col, const QString& text,
                       QSpinBox*& out, int lo, int hi, int val) {
        grid->addWidget(new QLabel(text), row, col * 2);
        out = new QSpinBox;
        out->setRange(lo, hi);
        out->setValue(val);
        grid->addWidget(out, row, col * 2 + 1);
    };

    addSpin(0, 0, QStringLiteral("训练样本数"), trainSpin_, 100, 42000, 10000);
    addSpin(0, 1, QStringLiteral("验证样本数"), valSpin_, 100, 10000, 2000);
    addSpin(0, 2, QStringLiteral("批大小"), batchSpin_, 1, 256, 32);
    addSpin(0, 3, QStringLiteral("轮数 (epochs)"), epochsSpin_, 1, 100, 5);

    grid->addWidget(new QLabel(QStringLiteral("学习率")), 1, 0);
    lrSpin_ = new QDoubleSpinBox;
    lrSpin_->setRange(0.0001, 1.0);
    lrSpin_->setDecimals(4);
    lrSpin_->setSingleStep(0.001);
    lrSpin_->setValue(0.01);
    grid->addWidget(lrSpin_, 1, 1);

    startBtn_ = new QPushButton(QStringLiteral("开始训练"));
    stopBtn_ = new QPushButton(QStringLiteral("停止"));
    stopBtn_->setEnabled(false);
    grid->addWidget(startBtn_, 1, 2);
    grid->addWidget(stopBtn_, 1, 3);

    root->addWidget(ctrl);

    // ---- 状态栏 ----
    statusLabel_ = new QLabel(QStringLiteral("就绪。点击「开始训练」。"));
    statusLabel_->setWordWrap(true);
    root->addWidget(statusLabel_);

    // ---- 中部：损失曲线 + 准确率曲线 + 样本预览 ----
    auto* plots = new QHBoxLayout;
    lossPlot_ = new PlotWidget;
    lossPlot_->setTitle(QStringLiteral("训练损失 (Loss)"));
    lossPlot_->setYLabel(QStringLiteral("loss"));
    lossPlot_->setXLabel(QStringLiteral("epoch"));
    accPlot_ = new PlotWidget;
    accPlot_->setTitle(QStringLiteral("验证准确率 (Accuracy)"));
    accPlot_->setYLabel(QStringLiteral("acc %"));
    accPlot_->setXLabel(QStringLiteral("epoch"));
    digit_ = new DigitWidget;

    plots->addWidget(lossPlot_);
    plots->addWidget(accPlot_);
    plots->addWidget(digit_);
    root->addLayout(plots, 1);

    connect(startBtn_, &QPushButton::clicked, this, &MainWindow::startTraining);
    connect(stopBtn_, &QPushButton::clicked, this, &MainWindow::stopTraining);
}

void MainWindow::startTraining() {
    startBtn_->setEnabled(false);
    stopBtn_->setEnabled(true);
    lossPlot_->clear();
    accPlot_->clear();
    statusLabel_->setText(QStringLiteral("正在启动训练线程..."));

    // 预览第一张训练样本（在界面线程加载，仅 1 行，很快）
    try {
        cnn::Dataset first = cnn::load_mnist_csv("data/train.csv", true, 1);
        if (first.num_samples() > 0) {
            QVector<float> px;
            px.reserve(784);
            for (float v : first.images[0]) px.append(v);
            digit_->setImage(px);
            digit_->setLabels(-1, int(first.labels[0]));
        }
    } catch (const std::exception&) {
        // 预览失败不影响训练
    }

    trainer_ = new Trainer;
    Trainer::Config cfg;
    cfg.trainSize = size_t(trainSpin_->value());
    cfg.valSize   = size_t(valSpin_->value());
    cfg.batchSize = size_t(batchSpin_->value());
    cfg.epochs    = epochsSpin_->value();
    cfg.lr        = float(lrSpin_->value());
    trainer_->setConfig(cfg);

    thread_ = new QThread(this);
    trainer_->moveToThread(thread_);

    connect(thread_, &QThread::started, trainer_, &Trainer::start);
    connect(trainer_, &Trainer::finished, thread_, &QThread::quit);
    connect(trainer_, &Trainer::finished, trainer_, &QObject::deleteLater);
    connect(thread_, &QThread::finished, thread_, &QObject::deleteLater);
    connect(trainer_, &Trainer::epochFinished, this, &MainWindow::onEpochFinished);
    connect(trainer_, &Trainer::logMessage, this, &MainWindow::onLogMessage);
    connect(trainer_, &Trainer::finished, this, &MainWindow::onTrainingFinished);

    thread_->start();
}

void MainWindow::stopTraining() {
    if (trainer_) trainer_->stop();  // 直接置原子标志位，worker 循环会检查并退出
    statusLabel_->setText(QStringLiteral("正在停止（等待当前批次结束）..."));
}

void MainWindow::onEpochFinished(int epoch, int total, double loss, double valAcc,
                                 double seconds, int pred, int truth) {
    lossPlot_->addPoint(loss);
    accPlot_->addPoint(valAcc * 100.0);
    digit_->setLabels(pred, truth);
    statusLabel_->setText(QStringLiteral("epoch %1/%2   loss= %3   val_acc= %4%  ( %5 s)")
                              .arg(epoch).arg(total)
                              .arg(loss, 0, 'f', 4)
                              .arg(valAcc * 100.0, 0, 'f', 2)
                              .arg(seconds, 0, 'f', 1));
}

void MainWindow::onLogMessage(const QString& msg) {
    statusLabel_->setText(msg);
}

void MainWindow::onTrainingFinished() {
    startBtn_->setEnabled(true);
    stopBtn_->setEnabled(false);
    statusLabel_->setText(statusLabel_->text() + QStringLiteral("   —— 训练结束。"));
}
