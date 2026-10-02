// include/cnn/relu.h
// ReLU 激活层：对输入逐元素做 y = max(0, x)。
//   - 无学习参数（没有权重/偏置/梯度），输入输出形状完全相同，属纯函数式层。
//   - 反向传播：dx = grad_output ⊙ (x > 0)，即在 x>0 处透传上游梯度、否则为 0；
//     x == 0 处按常用约定取导数为 0。
#pragma once

#include "cnn/tensor.h"

namespace cnn {

class ReLU {
public:
    using value_type = Tensor::value_type;  // 元素类型 float
    using size_type  = Tensor::size_type;   // 大小 / 坐标类型

    // 前向传播：逐元素 y = max(0, x)，输入输出形状相同。
    Tensor forward(const Tensor& input) const;

    // 反向传播：给定本次前向的输入 input 与上游梯度 grad_output（形状须与 input 一致），
    // 返回 dx = grad_output ⊙ (input > 0)。因 ReLU 无参数，故 backward 为 const 且无需累加梯度。
    Tensor backward(const Tensor& input, const Tensor& grad_output) const;

    // 本层无「关键参数」——既无形状超参，也无可学习权重；仅有上述两个纯函数式方法。
};

} // namespace cnn
