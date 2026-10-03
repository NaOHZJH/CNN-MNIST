// src/flatten.h

#include "cnn/flatten.h"

cnn::Tensor cnn::Flatten::forward(const Tensor& input) const {
    if (input.dim() < 2 || input.empty()) {
        throw std::invalid_argument("Input must have at least 2 dimensions and be non-empty for Flatten forward.");
    }

    size_type N = input.shape()[0];
    size_type F = input.size() / N;

    Tensor output = input;
    output.reshape({N, F});

    return output;
}

cnn::Tensor cnn::Flatten::backward(const Tensor& input, const Tensor& grad_output) const {
    if (grad_output.dim() != 2 || grad_output.size() != input.size()) {
        throw std::invalid_argument("grad_output must be 2D and have the same number of elements as input for Flatten backward.");
    }

    Tensor dx = grad_output;
    dx.reshape(input.shape());

    return dx;
}