// src/mnist_test.cpp
// 控制台训练/测试程序：加载 MNIST CSV，训练一个小型 CNN，报告每轮损失与验证准确率。
// 支持模型导出 / 加载（方案 A，见 include/cnn/model_io.h）：
//   （无参数）或 "train"         → 训练并保存到 model.bin
//   "train <out_path>"            → 训练并保存到指定路径
//   "load <模型文件>"              → 从文件加载参数，在验证集上评估准确率
// 运行方式（在项目根目录下）：./build/mnist_test.exe
#include <algorithm>  // std::shuffle / std::min
#include <chrono>
#include <exception>  // std::exception
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
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
#include "cnn/model_io.h"

using namespace cnn;

namespace {
constexpr size_t IMG_SIDE   = 28;
constexpr size_t IMG_PIXELS = IMG_SIDE * IMG_SIDE;   // 784

// ---- 可调参数（训练超参；load 模式只用到 kValSize / kBatchSize）----
constexpr size_t kTrainSize = 10000;   // 训练样本数（最多 42000）
constexpr size_t kValSize   = 2000;    // 验证样本数
constexpr size_t kBatchSize = 32;      // 批大小
constexpr int    kEpochs    = 5;       // 训练轮数
constexpr float  kLr        = 0.01f;   // 学习率
constexpr const char* kDefaultModelPath = "model.bin";  // 默认导出路径

// 构建网络：Conv->ReLU->Pool->Conv->ReLU->Pool->Flatten->Dense。
// 训练与加载必须使用同一结构（方案 A 约定），故单独抽出此函数供两处共用。
Network build_network() {
    Network net;
    net.add(Conv2D(1, 8, 3, 1, 1));     // [N,1,28,28] -> [N,8,28,28]
    net.add(ReLU());
    net.add(MaxPool2D(2, 2));           // -> [N,8,14,14]
    net.add(Conv2D(8, 16, 3, 1, 1));    // -> [N,16,14,14]
    net.add(ReLU());
    net.add(MaxPool2D(2, 2));           // -> [N,16,7,7]
    net.add(Flatten());                 // -> [N,784]
    net.add(Dense(16 * 7 * 7, 10));     // -> [N,10]
    return net;
}

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

// 从 train.csv 读取并切分出训练集 / 验证集（训练与加载共用同一份划分，保证结构一致）。
void load_split(const std::string& path, Dataset& train, Dataset& val) {
    Dataset full = load_mnist_csv(path, true, kTrainSize + kValSize);
    train.images.assign(full.images.begin(), full.images.begin() + kTrainSize);
    train.labels.assign(full.labels.begin(), full.labels.begin() + kTrainSize);
    val.images.assign(full.images.begin() + kTrainSize, full.images.end());
    val.labels.assign(full.labels.begin() + kTrainSize, full.labels.end());
}

// 训练并导出模型到 out_path
int run_train(const std::string& out_path) {
    // 1) 加载数据
    std::cout << "加载训练数据 data/train.csv ...\n";
    Dataset train, val;
    load_split("data/train.csv", train, val);
    std::cout << "  训练集 " << train.num_samples() << " / 验证集 " << val.num_samples() << " 个样本\n";

    // 2) 构建网络
    Network net = build_network();
    CrossEntropy loss;
    SGD sgd(kLr);
    Tensor().seed(12345);

    // 3) 训练
    std::vector<size_t> order(kTrainSize);
    for (size_t i = 0; i < kTrainSize; ++i) order[i] = i;
    std::mt19937 rng(0);

    std::cout << "开始训练（epochs=" << kEpochs << ", batch=" << kBatchSize << "）...\n";
    for (int ep = 1; ep <= kEpochs; ++ep) {
        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(order.begin(), order.end(), rng);

        float total_loss = 0.0f;
        size_t batches = 0;
        for (size_t s = 0; s + kBatchSize <= kTrainSize; s += kBatchSize) {
            std::vector<size_t> idx(order.begin() + s, order.begin() + s + kBatchSize);
            Tensor x = make_image_batch(train.images, idx);
            std::vector<size_t> y(kBatchSize);
            for (size_t i = 0; i < kBatchSize; ++i) y[i] = train.labels[idx[i]];

            net.zero_grad();
            Tensor out = net.forward(x);
            total_loss += loss.forward(out, y);
            Tensor g = loss.backward(out, y);   // dL/dout
            net.backward(g);                    // 反向传播到各层
            sgd.step(net);                      // 更新参数
            ++batches;
        }

        float acc = evaluate(net, val, 0, kValSize, kBatchSize);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "epoch " << ep << "/" << kEpochs
                  << "  loss = " << std::fixed << std::setprecision(4) << (total_loss / batches)
                  << "  val_acc = " << std::setprecision(2) << (acc * 100.0f) << "%"
                  << "  (" << std::setprecision(1) << secs << "s)\n";
    }

    // 4) 导出模型
    std::cout << "训练完成，导出模型到 " << out_path << " ...\n";
    save_model(net, out_path);
    std::cout << "已导出。\n";
    return 0;
}

// 从文件加载模型并在验证集上评估
int run_load(const std::string& in_path) {
    // 1) 加载验证数据（与训练时同一份划分）
    std::cout << "加载验证数据 data/train.csv ...\n";
    Dataset train, val;
    load_split("data/train.csv", train, val);
    std::cout << "  验证集 " << val.num_samples() << " 个样本\n";

    // 2) 构建与训练时一致的结构，再回填参数
    std::cout << "构建网络并加载参数 " << in_path << " ...\n";
    Network net = build_network();
    try {
        load_model(net, in_path);
    } catch (const std::exception& e) {
        std::cerr << "加载失败：" << e.what() << "\n";
        return 1;
    }

    // 3) 评估
    float acc = evaluate(net, val, 0, kValSize, kBatchSize);
    std::cout << "验证准确率 = " << std::fixed << std::setprecision(2)
              << (acc * 100.0f) << "%\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "train";

    if (mode == "load") {
        if (argc < 3) {
            std::cerr << "用法: mnist_test load <模型文件>\n";
            return 1;
        }
        return run_load(argv[2]);
    }

    if (mode != "train") {
        std::cerr << "未知模式: " << mode << "（可用: train [out_path] / load <in_path>）\n";
        return 1;
    }

    const std::string out_path = (argc > 2) ? argv[2] : kDefaultModelPath;
    return run_train(out_path);
}
