// src/pooling.cpp

#include "cnn/pooling.h"
#include <limits>
#include <algorithm>
#include <stdexcept>

cnn::MaxPool2D::MaxPool2D(size_type pool_size, size_type stride, size_type padding) : 
    pool_size_(pool_size), stride_(stride), padding_(padding) {
    if (pool_size_ == 0 || stride_ == 0) {
        throw std::invalid_argument("pool_size and stride must be positive");
    }
}

cnn::Tensor  cnn::MaxPool2D::forward(const Tensor& input) const {
    if (input.dim() != 4) {
        throw std::invalid_argument("Input must be a 4D tensor for MaxPool2D forward.");
    }
    size_type N = input.shape()[0];
    size_type C = input.shape()[1];
    size_type H_in = input.shape()[2];
    size_type W_in = input.shape()[3];

    // 计算输出形状
    if (H_in + 2 * padding_ < pool_size_ || W_in + 2 * padding_ < pool_size_) {
        throw std::invalid_argument("Input too small for the given pool_size and padding.");
    }
    size_type H_out = (H_in + 2 * padding_ - pool_size_) / stride_ + 1;
    size_type W_out = (W_in + 2 * padding_ - pool_size_) / stride_ + 1;

    Tensor output({N, C, H_out, W_out});

    for (size_type n = 0; n < N; ++n) {
        for (size_type c = 0; c < C; ++c) {
            for (size_type oh = 0; oh < H_out; ++oh) {
                for (size_type ow = 0; ow < W_out; ++ow) {
                    value_type max_val = std::numeric_limits<value_type>::lowest();
                    for (size_type kh = 0; kh < pool_size_; ++kh) {
                        for (size_type kw = 0; kw < pool_size_; ++kw) {
                            size_type ih_raw = oh * stride_ + kh;
                            size_type iw_raw = ow * stride_ + kw;
                            if (ih_raw < padding_ || iw_raw < padding_) continue;
                            size_type ih = ih_raw - padding_;
                            size_type iw = iw_raw - padding_;
                            if (ih >= H_in || iw >= W_in) continue;
                            max_val = std::max(max_val, input.at(n, c, ih, iw));
                        }
                    }
                    output.at(n, c, oh, ow) = max_val;
                }
            }
        }
    }
    return output;
}

cnn::Tensor cnn::MaxPool2D::backward(const Tensor& input, const Tensor& grad_output) const {
    if (input.dim() != 4 || grad_output.dim() != 4) {
        throw std::invalid_argument("Input and grad_output must be 4D tensors for MaxPool2D backward.");
    }
    size_type N = input.shape()[0];
    size_type C = input.shape()[1];
    size_type H_in = input.shape()[2];
    size_type W_in = input.shape()[3];
    size_type H_out = grad_output.shape()[2];
    size_type W_out = grad_output.shape()[3];

    Tensor dx({N, C, H_in, W_in});
    dx.fill(0); // 初始化梯度为 0

    for (size_type n = 0; n < N; ++n) {
        for (size_type c = 0; c < C; ++c) {
            for (size_type oh = 0; oh < H_out; ++oh) {
                for (size_type ow = 0; ow < W_out; ++ow) {
                    value_type max_val = std::numeric_limits<value_type>::lowest();
                    size_type max_ih = 0, max_iw = 0;
                    for (size_type kh = 0; kh < pool_size_; ++kh) {
                        for (size_type kw = 0; kw < pool_size_; ++kw) {
                            size_type ih_raw = oh * stride_ + kh;
                            size_type iw_raw = ow * stride_ + kw;
                            if (ih_raw < padding_ || iw_raw < padding_) continue;
                            size_type ih = ih_raw - padding_;
                            size_type iw = iw_raw - padding_;
                            if (ih >= H_in || iw >= W_in) continue;
                            value_type val = input.at(n, c, ih, iw);
                            if (val > max_val) {
                                max_val = val;
                                max_ih = ih;
                                max_iw = iw;
                            }
                        }
                    }
                    dx.at(n, c, max_ih, max_iw) += grad_output.at(n, c, oh, ow);
                }
            }
        }
    }
    return dx;
}