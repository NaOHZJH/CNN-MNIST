// include/cnn/pooling.h
// 最大池化层（MaxPool2D）：对 4D 输入在空间维（H、W）做滑动窗口取最大值，实现下采样。
//   输入 x：shape [N, C, H, W]
//   输出 y：shape [N, C, H_out, W_out]（通道数 C 不变，仅空间尺寸缩小）
//   输出尺寸：H_out = (H + 2*padding - pool_size) / stride + 1（整除下取整），W_out 同理。
//   - 无学习参数（没有权重/偏置/梯度），属纯函数式层（同 ReLU / Softmax）。
//   - 反向传播：把上游梯度路由到每个窗口内「最大值所在位置」，窗口内其余位置梯度为 0。
//   - pool_size / stride / padding 均按「正方形 / 对称」处理（单一整数值）。
#pragma once

#include "cnn/tensor.h"

namespace cnn {

class MaxPool2D {
public:
    using value_type = Tensor::value_type;  // 元素类型 float
    using size_type  = Tensor::size_type;   // 大小 / 坐标类型

    // 构造：指定池化窗口边长 pool_size、步长 stride、填充 padding。
    // 常用配置：pool_size=2、stride=2（经典 2×2 不重叠池化，空间尺寸减半）。
    explicit MaxPool2D(size_type pool_size, size_type stride, size_type padding = 0);

    // 前向传播：input[N, C, H, W] -> output[N, C, H_out, W_out]，每个窗口取最大值。
    // 会校验 input.dim()==4，否则抛异常。
    //   实现提示：先算 H_out/W_out（注意与 Conv2D 相同的无符号下溢问题——建议先校验
    //   H_in + 2*padding >= pool_size 等）；对每个输出位置在其窗口内遍历，取有效位置的
    //   最大值（padding 补齐的越界位置视为 -∞，不参与取 max）。
    Tensor forward(const Tensor& input) const;

    // 反向传播：给定本次前向的输入 input 与上游梯度 grad_output（形状同 output），
    // 返回 dx（形状同 input）。因 MaxPool 无参数，故 backward 为 const 且无需累加梯度。
    //   实现提示：对每个输出位置 (n, c, oh, ow)，在其窗口内重新找到最大值位置 (ih*, iw*)，
    //   仅把 grad_output[n,c,oh,ow] 加到 dx[n,c,ih*,iw*]；窗口内其余位置梯度为 0。
    //   （若有多个位置同取最大值，简单实现取「第一个」即可，对训练影响极小。）
    Tensor backward(const Tensor& input, const Tensor& grad_output) const;

    // ---- 超参访问 ----
    size_type pool_size() const noexcept { return pool_size_; }  // 池化窗口边长
    size_type stride() const noexcept { return stride_; }        // 步长
    size_type padding() const noexcept { return padding_; }      // 填充

    // 本层无「权重/偏置/梯度」成员——纯函数式，仅保存三个超参。

private:
    size_type pool_size_;  // 池化窗口边长
    size_type stride_;     // 步长
    size_type padding_;    // 填充
};

} // namespace cnn
