// src/conv2d.cpp

#include "cnn/conv2d.h"
#include <stdexcept> // std::invalid_argument
#include <cmath> // std::sqrt

cnn::Conv2D::Conv2D(size_type in_channels, size_type out_channels, 
                    size_type kernel_size, size_type stride, size_type padding) : 
                    in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
                    stride_(stride), padding_(padding) {
    // 初始化权重矩阵与偏执矩阵
    weight_ = Tensor({out_channels, in_channels, kernel_size, kernel_size});
    bias_ = Tensor({out_channels});
    // 权重矩阵采用 Xavier(Glorot) 均匀分布初始化
    value_type fan_in = in_channels * kernel_size * kernel_size;
    value_type fan_out = out_channels * kernel_size * kernel_size;
    value_type limit = std::sqrt(6.0 / (fan_in + fan_out));
    weight_.random_uniform(-limit, limit);
    bias_.fill(0.0f);
    // 初始化梯度矩阵
    grad_weight_ = Tensor::zeros({out_channels, in_channels, kernel_size, kernel_size});
    grad_bias_ = Tensor::zeros({out_channels});
}

cnn::Tensor cnn::Conv2D::forward(const Tensor& input) const {
    if (input.dim() != 4 || input.shape()[1] != in_channels_) {
        throw std::invalid_argument("Input shape mismatch for Conv2D layer.");
    }
    // 计算输出形状
    size_type N = input.shape()[0];
    size_type H_in = input.shape()[2];
    size_type W_in = input.shape()[3];
    size_type H_out = std::floor((H_in + 2 * padding_ - kernel_size_) / stride_) + 1;
    size_type W_out = std::floor((W_in + 2 * padding_ - kernel_size_) / stride_) + 1;
    Tensor output({N, out_channels_, H_out, W_out});
    if (H_in + 2 * padding_ < kernel_size_ || W_in + 2 * padding_ < kernel_size_)
        throw std::invalid_argument("Conv2D: input too small for kernel/padding.");

    // 滑窗卷积：对每个输出位置累加「输入窗口 · 权重」点积
    for (size_type n = 0; n < N; ++n) {
        for (size_type oc = 0; oc < out_channels_; ++oc) {
            for (size_type oh = 0; oh < H_out; ++oh) {
                for (size_type ow = 0; ow < W_out; ++ow) {
                    value_type acc = bias_.at(oc);   // 先加偏置
                    for (size_type ic = 0; ic < in_channels_; ++ic) {
                        for (size_type kh = 0; kh < kernel_size_; ++kh) {
                            // 行方向：ih = oh*stride + kh - padding，需 0 <= ih < H_in
                            size_type ih_raw = oh * stride_ + kh;   // 用非负中间量避免无符号下溢
                            if (ih_raw < padding_) continue;        // ih < 0（padding 补 0，跳过）
                            size_type ih = ih_raw - padding_;
                            if (ih >= H_in) continue;               // ih >= H（越界，跳过）

                            for (size_type kw = 0; kw < kernel_size_; ++kw) {
                                // 列方向：iw = ow*stride + kw - padding，需 0 <= iw < W_in
                                size_type iw_raw = ow * stride_ + kw;
                                if (iw_raw < padding_) continue;
                                size_type iw = iw_raw - padding_;
                                if (iw >= W_in) continue;

                                acc += input.at(n, ic, ih, iw) * weight_.at(oc, ic, kh, kw);
                            }
                        }
                    }
                    output.at(n, oc, oh, ow) = acc;
                }
            }
        }
    }
    return output;
}

cnn::Tensor cnn::Conv2D::backward(const Tensor& input, const Tensor& grad_output) {
    // 1) 形状校验：input 为 [N, C_in, H, W]；grad_output 为 [N, C_out, H_out, W_out]（batch 一致）
    if (input.dim() != 4 || input.shape()[1] != in_channels_) {
        throw std::invalid_argument("Input shape mismatch for Conv2D backward.");
    }
    if (grad_output.dim() != 4 || grad_output.shape()[1] != out_channels_ ||
        grad_output.shape()[0] != input.shape()[0]) {
        throw std::invalid_argument("grad_output shape mismatch for Conv2D backward.");
    }

    size_type N = input.shape()[0];
    size_type H_in = input.shape()[2];
    size_type W_in = input.shape()[3];
    size_type H_out = grad_output.shape()[2];
    size_type W_out = grad_output.shape()[3];

    // 2) 输入梯度 dx：先置零，随后通过「散播」累加（形状同 input）
    Tensor dx = Tensor::zeros({N, in_channels_, H_in, W_in});

    // 3) 一趟遍历同时得到三个梯度（g = grad_output 的单个元素）
    for (size_type n = 0; n < N; ++n) {
        for (size_type oc = 0; oc < out_channels_; ++oc) {
            for (size_type oh = 0; oh < H_out; ++oh) {
                for (size_type ow = 0; ow < W_out; ++ow) {
                    value_type g = grad_output.at(n, oc, oh, ow);

                    // 3a) 偏置梯度：dout 对 batch、空间维求和
                    grad_bias_.at(oc) += g;

                    for (size_type ic = 0; ic < in_channels_; ++ic) {
                        for (size_type kh = 0; kh < kernel_size_; ++kh) {
                            // 与 forward 相同的边界判断：ih = oh*stride + kh - padding
                            size_type ih_raw = oh * stride_ + kh;
                            if (ih_raw < padding_) continue;
                            size_type ih = ih_raw - padding_;
                            if (ih >= H_in) continue;

                            for (size_type kw = 0; kw < kernel_size_; ++kw) {
                                size_type iw_raw = ow * stride_ + kw;
                                if (iw_raw < padding_) continue;
                                size_type iw = iw_raw - padding_;
                                if (iw >= W_in) continue;

                                // 3b) 权重梯度：dout * 对应输入像素 input[n,ic,ih,iw]
                                grad_weight_.at(oc, ic, kh, kw) += g * input.at(n, ic, ih, iw);

                                // 3c) 输入梯度：dout * 对应权重，散播回输入位置 (ih,iw)
                                dx.at(n, ic, ih, iw) += g * weight_.at(oc, ic, kh, kw);
                            }
                        }
                    }
                }
            }
        }
    }

    return dx;
}

void cnn::Conv2D::zero_grad() {
    grad_weight_.fill(0.0f);
    grad_bias_.fill(0.0f);
}