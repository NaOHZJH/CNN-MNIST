// include/cnn/conv2d.h
// 二维卷积层：对 4D 输入做 valid/same 卷积（NCHW 布局）。
//   输入 x：shape [N, in_channels, H, W]
//   权重 W：shape [out_channels, in_channels, kernel_size, kernel_size]
//   偏置 b：shape [out_channels]（按输出通道广播）
//   输出 y：shape [N, out_channels, H_out, W_out]
//   输出尺寸：H_out = (H + 2*padding - kernel_size) / stride + 1（整除下取整），W_out 同理。
//   说明：kernel_size / stride / padding 均按「正方形 / 对称」处理（单一整数值）。
#pragma once

#include "cnn/tensor.h"

namespace cnn {

class Conv2D {
public:
    using value_type = Tensor::value_type;  // 元素类型 float
    using size_type  = Tensor::size_type;   // 大小 / 坐标类型

    // 构造：指定输入/输出通道数、卷积核边长、步长、填充。
    // 权重用 Xavier(Glorot) 均匀分布初始化：W ~ U(-a, a)，a = sqrt(6 / (fan_in + fan_out))，
    // 其中 fan_in = in_channels*k*k、fan_out = out_channels*k*k；偏置置 0。
    explicit Conv2D(size_type in_channels, size_type out_channels,
                    size_type kernel_size, size_type stride = 1, size_type padding = 0);

    // 前向传播：input[N, C_in, H, W] -> output[N, C_out, H_out, W_out]。
    // 会校验 input.dim()==4 且 input.shape()[1]==in_channels()，否则抛异常。
    Tensor forward(const Tensor& input) const;

    // 反向传播：给定本次前向的输入 input 与上游梯度 grad_output（形状同 output），
    // 把梯度【累加】进 grad_weight_ / grad_bias_，返回传给输入的梯度 dx（形状同 input）。
    //   实现提示（可用 im2col 展开后借助 matmul，或直接滑窗）：
    //     dx       = 把 weight 翻转后与 grad_output 做“full”卷积（转置卷积）
    //     dW[oc,ic,kh,kw] = Σ_{n,oh,ow} input[n,ic,oh*s+kh-p, ow*s+kw-p] * grad_output[n,oc,oh,ow]
    //     db[oc]  = Σ_{n,oh,ow} grad_output[n,oc,oh,ow]
    Tensor backward(const Tensor& input, const Tensor& grad_output);

    // ---- 参数访问 ----
    const Tensor& weight() const noexcept { return weight_; } // 权重 [out, in, k, k]
    Tensor& weight() noexcept { return weight_; }             // 权重（可变，供训练/加载用）
    const Tensor& bias() const noexcept { return bias_; }     // 偏置 [out_channels]
    Tensor& bias() noexcept { return bias_; }                 // 偏置（可变）

    // ---- 梯度访问（训练用）----
    const Tensor& grad_weight() const noexcept { return grad_weight_; } // 权重梯度 [out, in, k, k]
    Tensor& grad_weight() noexcept { return grad_weight_; }
    const Tensor& grad_bias() const noexcept { return grad_bias_; }     // 偏置梯度 [out_channels]
    Tensor& grad_bias() noexcept { return grad_bias_; }

    // 清零梯度：每次训练迭代前调用。
    void zero_grad();

    // ---- 超参访问 ----
    size_type in_channels() const noexcept { return in_channels_; }   // 输入通道数
    size_type out_channels() const noexcept { return out_channels_; } // 输出通道数
    size_type kernel_size() const noexcept { return kernel_size_; }   // 卷积核边长
    size_type stride() const noexcept { return stride_; }             // 步长
    size_type padding() const noexcept { return padding_; }           // 填充

private:
    size_type in_channels_;   // 输入通道数
    size_type out_channels_;  // 输出通道数
    size_type kernel_size_;   // 卷积核边长
    size_type stride_;        // 步长
    size_type padding_;       // 填充
    Tensor weight_;           // 权重 [out_channels, in_channels, kernel_size, kernel_size]
    Tensor bias_;             // 偏置 [out_channels]
    Tensor grad_weight_;      // 权重梯度（同 weight_ 形状）
    Tensor grad_bias_;        // 偏置梯度（同 bias_ 形状）
};

} // namespace cnn
