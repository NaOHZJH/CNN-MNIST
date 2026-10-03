// src/network.cpp

#include "cnn/network.h"

cnn::Tensor cnn::Network::forward(const Tensor& input) {
    cache_.clear();
    Tensor x = input;
    for (const auto& lw : layers_) {
        cache_.push_back(x);   // 记录本层输入，供 backward 重放
        x = lw.forward(x);     // 本层输出 -> 下一层输入
    }
    return x;
}

cnn::Tensor cnn::Network::backward(const Tensor& grad_output) {
    Tensor g = grad_output;
    // 逆序：从最后一层往回，用缓存的输入重放 backward(input, grad)
    for (size_type i = layers_.size(); i > 0; --i) {
        g = layers_[i - 1].backward(cache_[i - 1], g);
    }
    return g;
}

void cnn::Network::zero_grad() {
    for (auto& lw : layers_)
        lw.zero_grad();   // 无参层的 zero_grad 闭包为空操作
}

void cnn::Network::parameters(const std::function<void(Tensor&, Tensor&)>& visit) const {
    for (const auto& lw : layers_)
        lw.parameters(visit);   // 无参层不产出任何 (参数, 梯度) 对
}
