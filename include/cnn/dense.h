// include/cnn/dense.h
// 全连接层（Dense / Fully-Connected）：对 2D 输入做线性变换 y = x @ W + b。
//   输入 x：shape [N, in_features]
//   权重 W：shape [in_features, out_features]
//   偏置 b：shape [out_features]（按样本行广播相加）
//   输出 y：shape [N, out_features]
// 说明：这里 W 取 [in, out]（而非 PyTorch 的 [out, in]），因此可直接用 Tensor::matmul
//       实现 x @ W，无需转置。
#pragma once

#include "cnn/tensor.h"

namespace cnn {

class Dense {
public:
    using value_type = Tensor::value_type;  // 元素类型 float
    using size_type  = Tensor::size_type;   // 大小 / 坐标类型

    // 构造：指定输入/输出特征数；权重用 Xavier(Glorot) 均匀分布初始化，
    //       即 W ~ U(-a, a)，a = sqrt(6 / (in_features + out_features))；偏置置 0。
    explicit Dense(size_type in_features, size_type out_features);

    // 前向传播：input[N, in_features] -> output[N, out_features]。
    // 会校验 input.dim() == 2 且 input.shape()[1] == in_features()，否则抛异常。
    // 注意：偏置 b 为 [out_features]，需按行广播加到每个样本；
    //       Tensor 的同形元素运算不支持广播，实现时按行循环相加即可。
    Tensor forward(const Tensor& input) const;

    // 反向传播：给定本次前向的输入 input[N,in] 与上游梯度 grad_output[N,out]，
    // 计算并把梯度【累加】进 grad_weight_ / grad_bias_，返回传给输入的梯度 dx[N,in]。
    //   数学关系（W 为 [in,out]）：
    //     dx = grad_output @ W^T       （[N,out]×[out,in] → [N,in]，用 weight_.transpose()）
    //     dW = input^T @ grad_output   （[in,N]×[N,out] → [in,out]，用 input.transpose()）
    //     db = grad_output 沿 batch 维求和（[out]）
    Tensor backward(const Tensor& input, const Tensor& grad_output);

    // ---- 参数访问 ----
    const Tensor& weight() const noexcept { return weight_; } // 权重 [in_features, out_features]
    Tensor& weight() noexcept { return weight_; }             // 权重（可变，供训练/加载用）
    const Tensor& bias() const noexcept { return bias_; }     // 偏置 [out_features]
    Tensor& bias() noexcept { return bias_; }                 // 偏置（可变）

    // ---- 梯度访问（训练用）----
    const Tensor& grad_weight() const noexcept { return grad_weight_; } // 权重梯度 [in, out]
    Tensor& grad_weight() noexcept { return grad_weight_; }
    const Tensor& grad_bias() const noexcept { return grad_bias_; }     // 偏置梯度 [out]
    Tensor& grad_bias() noexcept { return grad_bias_; }

    // 清零梯度：每次训练迭代前调用，避免上一轮梯度累积。
    void zero_grad();

    // ---- 维度访问 ----
    size_type in_features() const noexcept { return in_features_; }   // 输入特征数
    size_type out_features() const noexcept { return out_features_; } // 输出特征数

private:
    size_type in_features_;   // 输入特征数
    size_type out_features_;  // 输出特征数
    Tensor weight_;           // 权重 [in_features, out_features]
    Tensor bias_;             // 偏置 [out_features]
    Tensor grad_weight_;      // 权重梯度 [in_features, out_features]
    Tensor grad_bias_;        // 偏置梯度 [out_features]
};

} // namespace cnn
