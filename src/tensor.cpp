// src/tensor.cpp
// Tensor 的实现。与 include/cnn/tensor.h 中的声明一一对应。
#include "cnn/tensor.h"

#include <algorithm>  // std::fill / std::max
#include <cmath>      // std::exp
#include <random>     // 随机数生成（引擎放在线程局部，见下）

namespace cnn {

namespace {

// 由 shape 计算元素总数；空 shape 返回 0。
Tensor::size_type numel(const std::vector<Tensor::size_type>& shape) {
    Tensor::size_type n = 1;
    for (Tensor::size_type d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

// 线程局部随机数引擎：默认用 std::random_device 播种一次，seed() 可覆盖。
std::mt19937& rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

} // namespace

// ---- 构造 ----
Tensor::Tensor(const std::vector<size_type>& shape, value_type fill_value)
    : shape_(shape) {
    compute_strides();
    data_.assign(numel(shape_), fill_value);
}

// ---- 形状信息 ----
Tensor::size_type Tensor::size() const noexcept {
    return numel(shape_);
}

// ---- 填充与随机初始化 ----
void Tensor::fill(value_type v) {
    std::fill(data_.begin(), data_.end(), v);
}

void Tensor::seed(unsigned int s) {
    rng().seed(s);
}

void Tensor::random_uniform(value_type low, value_type high) {
    std::uniform_real_distribution<value_type> dist(low, high);
    for (value_type& x : data_) x = dist(rng());
}

void Tensor::random_normal(value_type mean, value_type stddev) {
    std::normal_distribution<value_type> dist(mean, stddev);
    for (value_type& x : data_) x = dist(rng());
}

// ---- 变形 ----
Tensor& Tensor::reshape(const std::vector<size_type>& new_shape) {
    if (numel(new_shape) != size())
        throw std::invalid_argument("Tensor::reshape(): 元素总数不一致");
    shape_ = new_shape;
    compute_strides();
    return *this;
}

// ---- 元素级运算 ----
Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape_ != other.shape_)
        throw std::invalid_argument("Tensor::operator+=(): 形状不一致");
    for (size_type i = 0; i < size(); ++i) data_[i] += other.data_[i];
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (shape_ != other.shape_)
        throw std::invalid_argument("Tensor::operator-=(): 形状不一致");
    for (size_type i = 0; i < size(); ++i) data_[i] -= other.data_[i];
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if (shape_ != other.shape_)
        throw std::invalid_argument("Tensor::operator*=(): 形状不一致");
    for (size_type i = 0; i < size(); ++i) data_[i] *= other.data_[i];
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if (shape_ != other.shape_)
        throw std::invalid_argument("Tensor::operator/=(): 形状不一致");
    for (size_type i = 0; i < size(); ++i) data_[i] /= other.data_[i];
    return *this;
}

// ---- 矩阵 / 归约运算 ----
Tensor Tensor::matmul(const Tensor& other) const {
    if (dim() != 2 || other.dim() != 2)
        throw std::invalid_argument("Tensor::matmul(): 仅支持 2D");
    const size_type M = shape_[0], K = shape_[1], N = other.shape_[1];
    if (other.shape_[0] != K)
        throw std::invalid_argument("Tensor::matmul(): 内维不一致");

    // [M,K] 的行主序 strides 为 [K,1]，[K,N] 为 [N,1]，因此扁平下标如上计算。
    Tensor out({M, N}, value_type{0});
    for (size_type i = 0; i < M; ++i) {
        for (size_type j = 0; j < N; ++j) {
            value_type s = 0;
            for (size_type k = 0; k < K; ++k)
                s += data_[i * K + k] * other.data_[k * N + j];
            out.data_[i * N + j] = s;
        }
    }
    return out;
}

Tensor Tensor::transpose() const {
    if (dim() != 2)
        throw std::invalid_argument("Tensor::transpose(): 仅支持 2D");
    const size_type M = shape_[0], N = shape_[1];
    Tensor out({N, M});
    for (size_type i = 0; i < M; ++i)
        for (size_type j = 0; j < N; ++j)
            out.data_[j * M + i] = data_[i * N + j];
    return out;
}

Tensor Tensor::softmax(const Tensor& logits) {
    const auto& shape = logits.shape();
    if (shape.empty())
        throw std::invalid_argument("Tensor::softmax(): 空张量");
    const size_type C = shape.back();          // 最后一维大小（类别数）
    const size_type B = logits.size() / C;     // 前面的批量大小

    Tensor out(shape, value_type{0});
    for (size_type b = 0; b < B; ++b) {
        const size_type base = b * C;
        value_type mx = logits[base];
        for (size_type c = 1; c < C; ++c)      // 减去最大值，保证数值稳定
            mx = std::max(mx, logits[base + c]);

        value_type sum = 0;
        for (size_type c = 0; c < C; ++c) {
            const value_type e = std::exp(logits[base + c] - mx);
            out[base + c] = e;
            sum += e;
        }
        for (size_type c = 0; c < C; ++c)      // 归一化，使每行和为 1
            out[base + c] /= sum;
    }
    return out;
}

// ---- 便捷工厂 ----
Tensor Tensor::zeros(const std::vector<size_type>& shape) {
    return Tensor(shape, value_type{0});
}

Tensor Tensor::ones(const std::vector<size_type>& shape) {
    return Tensor(shape, value_type{1});
}

// ---- 逐元素二元运算（自由函数）----
Tensor operator+(const Tensor& a, const Tensor& b) { Tensor r = a; r += b; return r; }
Tensor operator-(const Tensor& a, const Tensor& b) { Tensor r = a; r -= b; return r; }
Tensor operator*(const Tensor& a, const Tensor& b) { Tensor r = a; r *= b; return r; }
Tensor operator/(const Tensor& a, const Tensor& b) { Tensor r = a; r /= b; return r; }

// ---- 私有辅助 ----
void Tensor::compute_strides() {
    const size_type nd = shape_.size();
    strides_.assign(nd, 0);
    size_type acc = 1;                          // 行主序：最后一维步长为 1
    for (size_type i = nd; i > 0; --i) {
        strides_[i - 1] = acc;
        acc *= shape_[i - 1];
    }
}

} // namespace cnn
