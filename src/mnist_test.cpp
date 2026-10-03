// src/mnist_test.cpp
// 控制台训练/测试程序：加载 MNIST CSV，训练一个小型 CNN，报告每轮损失与验证准确率。
// 运行方式（在项目根目录下）：./build/mnist_test.exe
#include <algorithm>  // std::shuffle / std::min
#include <chrono>
#include <iomanip>
#include <iostream>
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

using namespace cnn;

namespace {
constexpr size_t IMG_SIDE   = 28;
constexpr size_t IMG_PIXELS = IMG_SIDE * IMG_SIDE;   // 784

// 由平铺像素构造一个 batch 的 Tensor [B, 1, 28, 28]（NCHW，单通道）
Tensor make_image_batch(const std::vector<std::vector<float>>& images,
                        const std::vector<size_t>& idx) {
    const size_t B = idx.size();
    Tensor t({B, 1, IMG_SIDE, IMG_SIDE});
    for (size_t i = 0; i < B; ++i)
        for (size_t p = 0; p < IMG_PIXELS; ++p)
            t[i * IMG_PIXELS + p] = images[idx[i]][p];
    return t;
}

// 在数据集 [start, start+count) 区间上评估准确率（前向，不更新梯度）
float evaluate(Network& net, const Dataset& ds, size_t start, size_t count, size_t batch_size) {
    size_t correct = 0;
    for (size_t s = start; s < start + count; s += batch_size) {
        const size_t b = std::min(batch_size, start + count - s);
        std::vector<size_t> idx(b);
        for (size_t i = 0; i < b; ++i) idx[i] = s + i;

        Tensor out = net.forward(make_image_batch(ds.images, idx));
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
} // namespace

int main() {
    // ---- 可调参数 ----
    const size_t train_size = 10000;   // 用子集加速演示；想更准可调大（最多 42000）
    const size_t val_size   = 2000;
    const size_t batch_size = 32;
    const int    epochs     = 5;
    const float  lr         = 0.01f;

    // 1) 加载数据（只读前 train_size+val_size 行）
    std::cout << "加载训练数据 data/train.csv ...\n";
    Dataset full = load_mnist_csv("data/train.csv", true, train_size + val_size);
    std::cout << "  已加载 " << full.num_samples() << " 个样本\n";

    Dataset train, val;
    train.images.assign(full.images.begin(), full.images.begin() + train_size);
    train.labels.assign(full.labels.begin(), full.labels.begin() + train_size);
    val.images.assign(full.images.begin() + train_size, full.images.end());
    val.labels.assign(full.labels.begin() + train_size, full.labels.end());

    // 2) 构建网络：Conv->ReLU->Pool->Conv->ReLU->Pool->Flatten->Dense
    Network net;
    net.add(Conv2D(1, 8, 3, 1, 1));     // [N,1,28,28] -> [N,8,28,28]
    net.add(ReLU());
    net.add(MaxPool2D(2, 2));           // -> [N,8,14,14]
    net.add(Conv2D(8, 16, 3, 1, 1));    // -> [N,16,14,14]
    net.add(ReLU());
    net.add(MaxPool2D(2, 2));           // -> [N,16,7,7]
    net.add(Flatten());                 // -> [N,784]
    net.add(Dense(16 * 7 * 7, 10));     // -> [N,10]

    CrossEntropy loss;
    SGD sgd(lr);
    Tensor().seed(12345);

    // 3) 训练
    std::vector<size_t> order(train_size);
    for (size_t i = 0; i < train_size; ++i) order[i] = i;
    std::mt19937 rng(0);

    std::cout << "开始训练（epochs=" << epochs << ", batch=" << batch_size << "）...\n";
    for (int ep = 1; ep <= epochs; ++ep) {
        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(order.begin(), order.end(), rng);

        float total_loss = 0.0f;
        size_t batches = 0;
        for (size_t s = 0; s + batch_size <= train_size; s += batch_size) {
            std::vector<size_t> idx(order.begin() + s, order.begin() + s + batch_size);
            Tensor x = make_image_batch(train.images, idx);
            std::vector<size_t> y(batch_size);
            for (size_t i = 0; i < batch_size; ++i) y[i] = train.labels[idx[i]];

            net.zero_grad();
            Tensor out = net.forward(x);
            total_loss += loss.forward(out, y);
            Tensor g = loss.backward(out, y);   // dL/dout
            net.backward(g);                    // 反向传播到各层
            sgd.step(net);                      // 更新参数
            ++batches;
        }

        float acc = evaluate(net, val, 0, val_size, batch_size);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "epoch " << ep << "/" << epochs
                  << "  loss = " << std::fixed << std::setprecision(4) << (total_loss / batches)
                  << "  val_acc = " << std::setprecision(2) << (acc * 100.0f) << "%"
                  << "  (" << std::setprecision(1) << secs << "s)\n";
    }

    std::cout << "训练完成。\n";
    return 0;
}
