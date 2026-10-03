// include/cnn/flatten.h
// 展平层（Flatten）：把输入除 batch 维以外的所有维度压平成单个特征维，
// 用于在卷积/池化层之后、全连接层之前衔接。
//   输入 x：shape [N, ...]（任意 >= 2 维，通常为卷积输出的 [N, C, H, W]）
//   输出 y：shape [N, F]，其中 F = 除 batch 外各维大小的乘积（即 C*H*W）
//   - 无学习参数（同 ReLU / Softmax / MaxPool2D，属纯函数式层）。
//   - 展平是「纯 reshape」：NCHW 行主序下 [N,C,H,W] 与 [N,C*H*W] 的内存顺序完全一致，
//     因此无需重排数据，只需改变 shape（copy + reshape）。
//   - 反向传播：把 grad_output[N, F] reshape 回 input 的原 shape 即可。
#pragma once

#include "cnn/tensor.h"

namespace cnn {

class Flatten {
public:
    using value_type = Tensor::value_type;  // 元素类型 float
    using size_type  = Tensor::size_type;   // 大小 / 坐标类型

    // 前向传播：input[N, ...] -> output[N, F]，F = 除 batch 外各维的乘积。
    // 会校验 input.dim() >= 2（且非空），否则抛异常。
    //   实现提示：N = shape[0]，F = size() / N；输出 = 把 input 复制一份后 reshape({N, F})。
    Tensor forward(const Tensor& input) const;

    // 反向传播：给定本次前向的输入 input 与上游梯度 grad_output（形状 [N, F]），
    // 返回 dx（形状同 input）。因 Flatten 无参数，故 backward 为 const 且无需累加梯度。
    //   实现提示：校验 grad_output.dim()==2 且 grad_output.size()==input.size() 后，
    //   把 grad_output 复制一份 reshape 回 input.shape() 即可（展平不改变数据顺序）。
    Tensor backward(const Tensor& input, const Tensor& grad_output) const;

    // 本层无「关键参数」——既无形状超参，也无可学习权重；仅有上述两个纯函数式方法。
};

} // namespace cnn
