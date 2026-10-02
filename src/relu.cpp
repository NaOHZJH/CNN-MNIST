// src/relu.cpp

#include "cnn/relu.h"
#include <stdexcept> // std::invalid_argument

cnn::Tensor cnn::ReLU::forward(const Tensor& input) const{
    Tensor output(input);
    for (size_type i = 0; i < input.size(); ++i) {
        if (output[i] < 0) output[i] = 0;
    }
    return output;
}

cnn::Tensor cnn::ReLU::backward(const Tensor& input, const Tensor& grad_output) const{
    if (input.shape() != grad_output.shape()) {
        throw std::invalid_argument("Input and grad_output must have the same shape for ReLU backward.");
    }
    Tensor dx(input.shape());
    for (size_type i = 0; i < input.size(); ++i) {
        dx[i] = (input[i] > 0) ? grad_output[i] : 0;
    }
    return dx;
}