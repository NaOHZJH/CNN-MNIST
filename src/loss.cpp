// src/loss.cpp

#include "cnn/loss.h"
#include <cmath>
#include <numeric>
#include <algorithm>

cnn::CrossEntropy::value_type cnn::CrossEntropy::forward(const Tensor& pred, const std::vector<size_type>& target) const {
    if (pred.dim() != 2) {
        throw std::invalid_argument("Predictions must be a 2D tensor for CrossEntropy forward.");
    }
    size_type N = pred.shape()[0];
    size_type C = pred.shape()[1];
    if (target.size() != N) {
        throw std::invalid_argument("Target size must match the number of predictions for CrossEntropy forward.");
    }

    value_type loss = 0.0;
    for (size_type n = 0; n < N; ++n) {
        if (target[n] >= C) {
            throw std::out_of_range("Target index out of range for CrossEntropy forward.");
        }
        value_type m = *std::max_element(pred.data() + n * C, pred.data() + (n + 1) * C);
        value_type lse = m + std::log(std::accumulate(pred.data() + n * C, pred.data() + (n + 1) * C, 0.0, [m](value_type sum, value_type val) {
            return sum + std::exp(val - m);
        }));
        loss += lse - pred[n * C + target[n]];
    }
    return loss / N;
}

cnn::Tensor cnn::CrossEntropy::backward(const Tensor& pred, const std::vector<size_type>& target) const {
    if (pred.dim() != 2) {
        throw std::invalid_argument("Predictions must be a 2D tensor for CrossEntropy backward.");
    }
    size_type N = pred.shape()[0];
    size_type C = pred.shape()[1];
    if (target.size() != N) {
        throw std::invalid_argument("Target size must match the number of predictions for CrossEntropy backward.");
    }

    Tensor dx(pred.shape());
    for (size_type n = 0; n < N; ++n) {
        if (target[n] >= C) {
            throw std::out_of_range("Target index out of range for CrossEntropy backward.");
        }
        value_type m = *std::max_element(pred.data() + n * C, pred.data() + (n + 1) * C);
        value_type sum_exp = std::accumulate(pred.data() + n * C, pred.data() + (n + 1) * C, 0.0, [m](value_type sum, value_type val) {
            return sum + std::exp(val - m);
        });
        for (size_type c = 0; c < C; ++c) {
            value_type p_c = std::exp(pred[n * C + c] - m) / sum_exp;
            dx[n * C + c] = (p_c - (c == target[n] ? 1.0 : 0.0)) / N;
        }
    }
    return dx;
}