// gui/trainer.cpp

#include "trainer.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <random>
#include <vector>

#include "cnn/tensor.h"
#include "cnn/network.h"
#include "cnn/conv2d.h"
#include "cnn/relu.h"
#include "cnn/pooling.h"
#include "cnn/flatten.h"
#include "cnn/dense.h"
#include "cnn/loss.h"
#include "cnn/optimizer.h"
#include "cnn/dataset.h"

namespace {

constexpr size_t kImgSide   = 28;
constexpr size_t kImgPixels = kImgSide * kImgSide;  // 784

// 由平铺像素构造一个 batch 的 Tensor [B, 1, 28, 28]（NCHW，单通道）
cnn::Tensor makeImageBatch(const std::vector<std::vector<float>>& images,
                           const std::vector<size_t>& idx) {
    const size_t B = idx.size();
    cnn::Tensor t({B, 1, kImgSide, kImgSide});
    for (size_t i = 0; i < B; ++i)
        for (size_t p = 0; p < kImgPixels; ++p)
            t[i * kImgPixels + p] = images[idx[i]][p];
    return t;
}

// 在 [start, start+count) 区间上评估准确率（前向，不更新梯度）
float evaluate(cnn::Network& net, const cnn::Dataset& ds,
               size_t start, size_t count, size_t batchSize) {
    size_t correct = 0;
    for (size_t s = start; s < start + count; s += batchSize) {
        const size_t b = std::min(batchSize, start + count - s);
        std::vector<size_t> idx(b);
        for (size_t i = 0; i < b; ++i) idx[i] = s + i;

        cnn::Tensor out = net.forward(makeImageBatch(ds.images, idx));
        const size_t C = out.shape()[1];
        for (size_t n = 0; n < b; ++n) {
            size_t best = 0;
            float bestv = out.at(n, 0);
            for (size_t c = 1; c < C; ++c)
                if (out.at(n, c) > bestv) { bestv = out.at(n, c); best = c; }
            if (best == ds.labels[s + n]) ++correct;
        }
    }
    return float(correct) / float(count);
}

// 预测单个样本的类别
int predict(cnn::Network& net, const std::vector<std::vector<float>>& images, size_t idx) {
    std::vector<size_t> i{idx};
    cnn::Tensor out = net.forward(makeImageBatch(images, i));
    const size_t C = out.shape()[1];
    size_t best = 0;
    float bestv = out.at(0, 0);
    for (size_t c = 1; c < C; ++c)
        if (out.at(0, c) > bestv) { bestv = out.at(0, c); best = c; }
    return int(best);
}

} // namespace

Trainer::Trainer(QObject* parent) : QObject(parent) {}

void Trainer::setConfig(const Config& c) { cfg_ = c; }

void Trainer::start() {
    stopRequested_ = false;

    emit logMessage(QStringLiteral("正在加载数据 %1 ...").arg(cfg_.dataPath));
    cnn::Dataset full;
    try {
        full = cnn::load_mnist_csv(cfg_.dataPath.toStdString(), true,
                                   cfg_.trainSize + cfg_.valSize);
    } catch (const std::exception& e) {
        emit logMessage(QStringLiteral("加载失败：%1").arg(e.what()));
        emit finished();
        return;
    }

    const size_t trainSize = std::min(cfg_.trainSize, full.num_samples());
    const size_t valSize   = std::min(cfg_.valSize, full.num_samples() - trainSize);
    if (trainSize == 0 || valSize == 0) {
        emit logMessage(QStringLiteral("数据不足（train=%1 val=%2）").arg(trainSize).arg(valSize));
        emit finished();
        return;
    }
    emit logMessage(QStringLiteral("已加载 %1 个样本（train=%2 val=%3）")
                        .arg(full.num_samples()).arg(trainSize).arg(valSize));

    cnn::Dataset train, val;
    train.images.assign(full.images.begin(), full.images.begin() + trainSize);
    train.labels.assign(full.labels.begin(), full.labels.begin() + trainSize);
    val.images.assign(full.images.begin() + trainSize, full.images.end());
    val.labels.assign(full.labels.begin() + trainSize, full.labels.end());

    emit logMessage(QStringLiteral("构建网络：Conv->ReLU->Pool->Conv->ReLU->Pool->Flatten->Dense"));
    cnn::Network net;
    net.add(cnn::Conv2D(1, 8, 3, 1, 1));    // [N,1,28,28] -> [N,8,28,28]
    net.add(cnn::ReLU());
    net.add(cnn::MaxPool2D(2, 2));          // -> [N,8,14,14]
    net.add(cnn::Conv2D(8, 16, 3, 1, 1));   // -> [N,16,14,14]
    net.add(cnn::ReLU());
    net.add(cnn::MaxPool2D(2, 2));          // -> [N,16,7,7]
    net.add(cnn::Flatten());                // -> [N,784]
    net.add(cnn::Dense(16 * 7 * 7, 10));    // -> [N,10]

    cnn::CrossEntropy loss;
    cnn::SGD sgd(cfg_.lr);
    cnn::Tensor().seed(12345);

    std::vector<size_t> order(trainSize);
    for (size_t i = 0; i < trainSize; ++i) order[i] = i;
    std::mt19937 rng(0);

    const size_t batchSize = cfg_.batchSize;

    for (int ep = 1; ep <= cfg_.epochs; ++ep) {
        if (stopRequested_) break;
        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(order.begin(), order.end(), rng);

        float totalLoss = 0.0f;
        size_t batches = 0;
        for (size_t s = 0; s + batchSize <= trainSize; s += batchSize) {
            if (stopRequested_) break;
            std::vector<size_t> idx(order.begin() + s, order.begin() + s + batchSize);
            cnn::Tensor x = makeImageBatch(train.images, idx);
            std::vector<size_t> y(batchSize);
            for (size_t i = 0; i < batchSize; ++i) y[i] = train.labels[idx[i]];

            net.zero_grad();
            cnn::Tensor out = net.forward(x);
            totalLoss += loss.forward(out, y);
            cnn::Tensor g = loss.backward(out, y);
            net.backward(g);
            sgd.step(net);
            ++batches;
        }

        if (stopRequested_) break;

        double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();
        float valAcc = evaluate(net, val, 0, valSize, batchSize);
        int pred = predict(net, train.images, 0);   // 预览第一个训练样本的预测
        int truth = int(train.labels[0]);
        double avgLoss = batches ? totalLoss / batches : 0.0;

        emit logMessage(QStringLiteral("epoch %1/%2  损失= %3  验证准确率= %4%  ( %5 s)")
                            .arg(ep).arg(cfg_.epochs)
                            .arg(avgLoss, 0, 'f', 4)
                            .arg(valAcc * 100.0, 0, 'f', 2)
                            .arg(secs, 0, 'f', 1));
        emit epochFinished(ep, cfg_.epochs, avgLoss, valAcc, secs, pred, truth);
    }

    emit finished();
}

void Trainer::stop() { stopRequested_ = true; }
