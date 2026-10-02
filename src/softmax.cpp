// src/softmax.cpp

#include "cnn/softmax.h"
#include <stdexcept> // std::invalid_argument

cnn::Tensor cnn::Softmax::forward(const Tensor& input) const {
    return Tensor::softmax(input);
}

cnn::Tensor cnn::Softmax::backward(const Tensor& input, const Tensor& grad_output) const {
    if (input.shape() != grad_output.shape()) {
        throw std::invalid_argument("Input and grad_output must have the same shape for Softmax backward.");
    }
    if (input.empty()) {
        throw std::invalid_argument("Softmax backward: empty input.");
    }

    Tensor y = Tensor::softmax(input);
    Tensor dx(input.shape());
    size_type num_classes = input.shape().back();          // 最后一维 = 类别数
    size_type num_samples = input.size() / num_classes;    // 前面所有维 = batch 数

    for (size_type sample = 0; sample < num_samples; ++sample) {
        value_type s = 0;
        for (size_type c = 0; c < num_classes; ++c) {
            s += grad_output[sample * num_classes + c] * y[sample * num_classes + c];
        }
        for (size_type c = 0; c < num_classes; ++c) {
            dx[sample * num_classes + c] = y[sample * num_classes + c] * (grad_output[sample * num_classes + c] - s);
        }
    }

    return dx;
}