// include/cnn/network.h
// 顺序网络容器（Network）：把类型各异的层（Conv2D/Dense 有参数，ReLU/Softmax/MaxPool2D/Flatten 无参数）
// 统一存放，并按顺序驱动前向 / 反向传播。
//   - 类型擦除：每层被包装成 4 个 std::function 闭包（forward / backward / zero_grad / parameters），
//     从而无需为各层引入公共基类。
//   - 前向缓存：所有层的 backward 签名都是 backward(input, grad_output)，需要「该层前向时的输入」，
//     因此 forward 会缓存每层输入，backward 再逆序重放。
//   - 参数暴露：parameters(visit) 把每个含参层的 (weight, grad_weight)、(bias, grad_bias) 交给回调，
//     供优化器统一更新。无参层的 zero_grad / parameters 通过 if constexpr 在编译期跳过。
#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "cnn/tensor.h"

namespace cnn {

// ---- 类型特征：区分「有参层」与「无参层」----

// 是否有 zero_grad() 成员（Dense / Conv2D 有；ReLU/Softmax/MaxPool2D/Flatten 没有）
template <class T, class = void>
struct has_zero_grad : std::false_type {};
template <class T>
struct has_zero_grad<T, std::void_t<decltype(std::declval<T&>().zero_grad())>> : std::true_type {};

// 是否有 weight() 成员（Dense / Conv2D 有；无参层没有）
template <class T, class = void>
struct has_weight : std::false_type {};
template <class T>
struct has_weight<T, std::void_t<decltype(std::declval<T&>().weight())>> : std::true_type {};

class Network {
public:
    using value_type = Tensor::value_type;  // 元素类型 float
    using size_type  = Tensor::size_type;   // 大小 / 坐标类型

    // 加入一层：模板，任意具备 forward/backward 的层都能加入（按值拷贝保存）。
    // 用 shared_ptr 保存，是因为 backward / zero_grad 是非 const 成员函数——
    // 若直接按值捕获进（默认 const 的）lambda 会无法调用；shared_ptr 指向非 const 对象即可规避。
    template <class L>
    void add(L layer) {
        auto p = std::make_shared<L>(std::move(layer));
        layers_.push_back(LayerWrap{
            // 前向：input -> output
            [p](const Tensor& x) { return p->forward(x); },
            // 反向：给定本层输入与上游梯度，返回下游梯度
            [p](const Tensor& x, const Tensor& g) { return p->backward(x, g); },
            // 清零梯度：无参层在编译期跳过（空操作）
            [p]() {
                if constexpr (has_zero_grad<L>::value)
                    p->zero_grad();
            },
            // 遍历 (参数, 梯度) 对：无参层不产出任何对
            [p](const std::function<void(Tensor&, Tensor&)>& visit) {
                if constexpr (has_weight<L>::value) {
                    visit(p->weight(), p->grad_weight());
                    visit(p->bias(),   p->grad_bias());
                }
            },
        });
    }

    // 前向传播：依次把输入送过每一层，返回最终输出。
    // 同时缓存每层输入到 cache_（供 backward 重放），因此必须在 backward 之前调用。
    Tensor forward(const Tensor& input);

    // 反向传播：给定最终输出的上游梯度 grad_output，逆序逐层回传，返回对原始输入的梯度。
    // 需要此前已调用过 forward（否则 cache_ 为空，会越界）。
    Tensor backward(const Tensor& grad_output);

    // 清零所有含参层的梯度：每次训练迭代、做反向传播之前调用。
    void zero_grad();

    // 遍历网络内所有 (参数, 梯度) 对并交给回调 visit（例如 SGD 的 w -= lr*gw）。
    // 顺序约定：对每个含参层先 weight/grad_weight，再 bias/grad_bias。
    void parameters(const std::function<void(Tensor&, Tensor&)>& visit) const;

private:
    // 类型擦除后的「统一层」：用 4 个闭包抹平各层类型差异。
    struct LayerWrap {
        std::function<Tensor(const Tensor&)>                      forward;   // 前向
        std::function<Tensor(const Tensor&, const Tensor&)>       backward;  // 反向
        std::function<void()>                                     zero_grad; // 清零梯度
        std::function<void(const std::function<void(Tensor&, Tensor&)>&)> parameters; // 参数遍历
    };

    std::vector<LayerWrap> layers_;  // 按加入顺序存放的层
    std::vector<Tensor>    cache_;   // 每层输入的快照（forward 时填充，backward 时消费）
};

} // namespace cnn
