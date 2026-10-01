// include/cnn/tensor.h
// 多维张量：CNN 的基础数据结构。以 NCHW 布局存储，底层是一段连续的一维数组，
// 并预计算行主序 strides，从而把多维坐标 O(1) 映射到线性下标。
#pragma once

#include <cstddef>      // std::size_t
#include <type_traits>  // std::is_integral_v（at() 模板约束）
#include <stdexcept>    // std::out_of_range（at() 越界时抛出）
#include <vector>

namespace cnn {

class Tensor {
public:
    using value_type = float;        // 元素类型
    using size_type  = std::size_t;  // 大小 / 形状 / 坐标的类型

    // ---- 构造 ----
    Tensor() = default;                                   // 空张量：shape 为空，size() == 0
    explicit Tensor(const std::vector<size_type>& shape,  // 按 shape 构造并全部填充 fill_value
                    value_type fill_value = value_type{0});

    // ---- 形状信息 ----
    const std::vector<size_type>& shape() const noexcept { return shape_; } // 各维大小
    size_type dim() const noexcept { return shape_.size(); }               // 维度数（ndim）
    size_type size() const noexcept;                                        // 元素总数（空 shape 为 0）
    bool empty() const noexcept { return data_.empty(); }                   // 是否为空

    // ---- 数据访问 ----
    value_type* data() noexcept { return data_.data(); }             // 底层连续缓冲区首指针（可变）
    const value_type* data() const noexcept { return data_.data(); } // 底层连续缓冲区首指针（只读）
    value_type& operator[](size_type i) { return data_[i]; }             // 线性访问，不做越界检查
    const value_type& operator[](size_type i) const { return data_[i]; }

    // 按坐标访问（任意维，通常 NCHW）。索引个数必须等于 dim()，任一越界抛 std::out_of_range。
    template <typename... Indices>
    value_type& at(Indices... idx);
    template <typename... Indices>
    const value_type& at(Indices... idx) const;

    // ---- 填充与随机初始化 ----
    void fill(value_type v);                                // 全部元素置为 v
    void seed(unsigned int s);                              // 设置随机种子，使 random_* 结果可复现
    void random_uniform(value_type low, value_type high);   // 逐元素取 U[low, high) 均匀分布
    void random_normal(value_type mean, value_type stddev); // 逐元素取 N(mean, stddev) 正态分布

    // ---- 变形 ----
    Tensor& reshape(const std::vector<size_type>& new_shape); // 改变形状（元素总数须一致），返回 *this 支持链式

    // ---- 元素级运算（要求与 *this 形状一致，逐元素进行）----
    Tensor& operator+=(const Tensor& other); // 逐元素加
    Tensor& operator-=(const Tensor& other); // 逐元素减
    Tensor& operator*=(const Tensor& other); // 逐元素乘（哈达玛积）
    Tensor& operator/=(const Tensor& other); // 逐元素除

    // ---- 矩阵 / 归约运算 ----
    Tensor matmul(const Tensor& other) const;    // 矩阵乘法，仅支持 2D：[M,K]×[K,N] → [M,N]
    Tensor transpose() const;                    // 转置，仅支持 2D：[M,N] → [N,M]
    static Tensor softmax(const Tensor& logits); // 沿最后一维 softmax（数值稳定，通常用于 [N,Classes]）

    // ---- 便捷工厂 ----
    static Tensor zeros(const std::vector<size_type>& shape); // 构造全 0 张量
    static Tensor ones(const std::vector<size_type>& shape);  // 构造全 1 张量

private:
    std::vector<size_type> shape_;    // 各维大小
    std::vector<size_type> strides_;  // 行主序步长：linear = sum(idx_i * strides_[i])
    std::vector<value_type> data_;    // 连续存储的元素

    void compute_strides();            // 由 shape_ 重算 strides_
};

// ---- 逐元素二元运算（产生新张量，要求两侧形状一致）----
Tensor operator+(const Tensor& a, const Tensor& b); // 逐元素加
Tensor operator-(const Tensor& a, const Tensor& b); // 逐元素减
Tensor operator*(const Tensor& a, const Tensor& b); // 逐元素乘
Tensor operator/(const Tensor& a, const Tensor& b); // 逐元素除

// ================= 模板实现（需在头文件内可见，不能放到 .cpp）=================

template <typename... Indices>
Tensor::value_type& Tensor::at(Indices... idx) {
    return const_cast<value_type&>(static_cast<const Tensor&>(*this).at(idx...));
}

template <typename... Indices>
const Tensor::value_type& Tensor::at(Indices... idx) const {
    static_assert((std::is_integral_v<Indices> && ...),
                  "Tensor::at() 的索引必须是整数类型");
    if (sizeof...(Indices) != dim())
        throw std::out_of_range("Tensor::at(): 索引个数与 dim() 不一致");
    size_type off = 0, i = 0;
    auto acc = [&](size_type v) {
        if (v >= shape_[i])
            throw std::out_of_range("Tensor::at(): 索引越界");
        off += v * strides_[i];
        ++i;
    };
    (acc(static_cast<size_type>(idx)), ...); // 依次对每个坐标做边界检查并累加线性偏移
    return data_[off];
}

} // namespace cnn
