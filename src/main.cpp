// src/main.cpp
// 前向传播验证：搭一个简单 CNN（Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> Softmax），
// 用随机权重对随机输入跑通前向，打印每层输出 shape，并校验 Softmax 输出按行求和为 1。
#include <iostream>
#include <iomanip>
#include <cstddef>
#include <cmath>  // std::fabs

#include "cnn/tensor.h"
#include "cnn/conv2d.h"
#include "cnn/relu.h"
#include "cnn/pooling.h"
#include "cnn/dense.h"
#include "cnn/softmax.h"

using namespace cnn;

// 打印张量名称与 shape
static void print_shape(const char* name, const Tensor& t) {
    std::cout << std::left << std::setw(10) << name << " : [";
    const auto& s = t.shape();
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << s[i];
    }
    std::cout << "]\n";
}

int main() {
    // 固定随机种子（seed 作用于线程局部 RNG，所有层的 Xavier 初始化共用）
    Tensor().seed(42);

    // ---- 1) 输入：[N=2, C=1, H=8, W=8] ----
    Tensor input({2, 1, 8, 8});
    input.random_uniform(0.0f, 1.0f);
    print_shape("input", input);

    // ---- 2) 各层 ----
    Conv2D    conv(1, 4, 3, /*stride=*/1, /*padding=*/0);   // [2,1,8,8] -> [2,4,6,6]
    ReLU      relu;
    MaxPool2D pool(2, 2);                                  // [2,4,6,6] -> [2,4,3,3]
    Dense     dense(36, 10);                               // [2,36]    -> [2,10]
    Softmax   softmax;

    // ---- 3) 前向传播 ----
    Tensor x = conv.forward(input);
    print_shape("conv", x);

    x = relu.forward(x);
    print_shape("relu", x);

    x = pool.forward(x);
    print_shape("pool", x);

    // 展平：把 [N,C,H,W] -> [N, C*H*W]，再送入全连接层
    const std::size_t N = x.shape()[0];
    const std::size_t C = x.shape()[1];
    const std::size_t H = x.shape()[2];
    const std::size_t W = x.shape()[3];
    x.reshape({N, C * H * W});
    print_shape("flatten", x);

    x = dense.forward(x);
    print_shape("dense", x);

    x = softmax.forward(x);
    print_shape("softmax", x);

    // ---- 4) 校验：Softmax 每行求和应等于 1，并打印预测类别 ----
    const std::size_t num_samples = x.shape()[0];
    const std::size_t num_classes = x.shape()[1];
    bool ok = true;
    std::cout << "\nSoftmax 校验与预测:\n";
    for (std::size_t n = 0; n < num_samples; ++n) {
        float sum = 0.0f;
        std::size_t best = 0;
        float best_val = x.at(n, 0);
        for (std::size_t c = 0; c < num_classes; ++c) {
            const float v = x.at(n, c);
            sum += v;
            if (v > best_val) { best_val = v; best = c; }
        }
        std::cout << "  样本 " << n << ": 行和 = " << sum
                  << " (应≈1), 预测类别 = " << best << "\n";
        if (std::fabs(sum - 1.0f) > 1e-4f) ok = false;
    }

    if (ok) {
        std::cout << "\n== 前向传播验证通过：shape 链正确，Softmax 归一化正确 ==\n";
        return 0;
    } else {
        std::cout << "\n== 前向传播校验失败 ==\n";
        return 1;
    }
}
