// src/dense.cpp

#include "cnn/dense.h"
#include "cnn/tensor.h"
#include <cmath> // std::sqrt

cnn::Dense::Dense(cnn::Dense::size_type in_features, 
                  cnn::Dense::size_type out_features)
                  : in_features_(in_features), out_features_(out_features){
    weight_ = cnn::Tensor({in_features, out_features});
    bias_ = cnn::Tensor({out_features});
    grad_weight_ = cnn::Tensor::zeros({in_features, out_features});
    grad_bias_ = cnn::Tensor::zeros({out_features});

    // Xavier(Glorot) 均匀分布初始化权重
    float a = std::sqrt(6.0f / (in_features + out_features));
    weight_.random_uniform(-a, a);
    bias_.fill(0.0f);
}

cnn::Tensor cnn::Dense::forward(const cnn::Tensor& input) const {
    if (input.dim() != 2 || input.shape()[1] != in_features_) {
        throw std::invalid_argument("Input shape mismatch for Dense layer.");
    }
    cnn::Tensor output = input.matmul(weight_);
    for (cnn::Dense::size_type i = 0; i < output.shape()[0]; ++i) {
        for (cnn::Dense::size_type j = 0; j < out_features_; ++j) {
            output.at(i, j) += bias_.at(j);
        }
    }
    return output;
}

cnn::Tensor cnn::Dense::backward(const cnn::Tensor& input, const cnn::Tensor& grad_output) {
    if (input.dim() != 2 || input.shape()[1] != in_features_ ||
        grad_output.dim() != 2 || grad_output.shape()[1] != out_features_) {
        throw std::invalid_argument("Input or grad_output shape mismatch for Dense layer.");
    }

    // 计算梯度
    cnn::Tensor dx = grad_output.matmul(weight_.transpose());
    cnn::Tensor dW = input.transpose().matmul(grad_output);
    cnn::Tensor db = cnn::Tensor::zeros({out_features_});
    for (cnn::Dense::size_type i = 0; i < grad_output.shape()[0]; ++i) {
        for (cnn::Dense::size_type j = 0; j < out_features_; ++j) {
            db.at(j) += grad_output.at(i, j);
        }
    }

    // 累加梯度
    grad_weight_ += dW;
    grad_bias_ += db;

    return dx;
}

void cnn::Dense::zero_grad() {
    grad_weight_.fill(0.0f);
    grad_bias_.fill(0.0f);
}