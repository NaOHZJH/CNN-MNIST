// include/cnn/softmax.h
// Softmax 激活层：沿最后一维做数值稳定的 softmax，把每个样本的得分归一化为概率分布。
//   输入 x：shape [..., C]（最后一维 C 为类别数，通常 [N, C]）
//   输出 y：shape 同输入，每个样本沿最后一维求和为 1。
//   - 无学习参数（同 ReLU，属纯函数式层）。
//   - 反向传播：给定 g = dL/dy，dx_i = y_i * (g_i - Σ_j g_j * y_j)。
#pragma once

#include "cnn/tensor.h"

namespace cnn {

class Softmax {
public:
    using value_type = Tensor::value_type;  // 元素类型 float
    using size_type  = Tensor::size_type;   // 大小 / 坐标类型

    // 前向传播：沿最后一维做 softmax，输入输出形状相同。
    // 支持 2D [N, C]、1D [C]，或更高维（前面所有维视作 batch，最后一维为类别）。
    Tensor forward(const Tensor& input) const;

    // 反向传播：给定本次前向的输入 input 与上游梯度 grad_output（形状须与 input 一致），
    // 返回 dx。实现时内部重算 y = softmax(input)，再对每个样本沿最后一维套公式：
    //   s  = Σ_j g_j * y_j
    //   dx_i = y_i * (g_i - s)
    // 因 Softmax 无参数，故 backward 为 const 且无需累加梯度。
    Tensor backward(const Tensor& input, const Tensor& grad_output) const;

    // 本层无「关键参数」——无权重/偏置/梯度；唯一隐含约定是沿最后一维归一化。
};

} // namespace cnn
