// src/optimizer.cpp

#include "cnn/optimizer.h"

cnn::SGD::SGD(value_type learning_rate) : learning_rate_(learning_rate) {
    if (learning_rate_ <= 0) {
        throw std::invalid_argument("Learning rate must be positive.");
    }
}

void cnn::SGD::step(Network& net) const {
    net.parameters([this](Tensor& w, Tensor& gw) {
        for (size_type i = 0; i < w.size(); ++i) {
            w[i] -= learning_rate_ * gw[i];
        }
    });
}