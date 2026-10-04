// gui/trainer.h
// 后台训练器：加载 MNIST 数据、构建网络、训练，并通过信号把进度回报给界面线程。
// 本对象通过 moveToThread 放到 QThread 中运行，训练循环在 start() 里同步执行，
// 因此界面不会卡顿。
#pragma once

#include <QObject>
#include <QString>

#include <atomic>
#include <cstddef>

class Trainer : public QObject {
    Q_OBJECT
public:
    struct Config {
        QString dataPath = QStringLiteral("data/train.csv");
        size_t trainSize = 10000;
        size_t valSize   = 2000;
        size_t batchSize = 32;
        int    epochs    = 5;
        float  lr        = 0.01f;
        QString modelPath = QStringLiteral("model.bin");  // 训练结束后导出；空串则不导出
    };

    explicit Trainer(QObject* parent = nullptr);
    void setConfig(const Config& cfg);

public slots:
    void start();  // 在 worker 线程中调用：加载数据并开始训练
    void stop();   // 请求停止（线程安全，仅置原子标志位）

signals:
    void logMessage(const QString& msg);
    // 每个 epoch 结束后发出：epoch、总轮数、平均损失、验证准确率(0..1)、耗时秒数、样本预测/真实类别
    void epochFinished(int epoch, int totalEpochs, double loss, double valAcc,
                       double seconds, int predLabel, int trueLabel);
    void finished();

private:
    Config cfg_;
    std::atomic<bool> stopRequested_{false};
};
