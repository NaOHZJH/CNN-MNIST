// include/cnn/optimizer.h
// 随机梯度下降优化器（SGD）：持有一个学习率，对网络所有参数做 w -= lr * grad。
//   - 用法：每次迭代先 net.zero_grad()，再前向 + 反向得到各层梯度，最后 sgd.step(net) 更新。
//   - step 内部不负责清零梯度（清零由 Network::zero_grad() 在反向传播前完成）。
//   - 因 Tensor 暂无标量乘法，更新采用逐元素循环：w[i] -= lr * gw[i]。
#pragma once

#include "cnn/tensor.h"
#include "cnn/network.h"

namespace cnn {

class SGD {
public:
    using value_type = Tensor::value_type;  // 元素类型 float
    using size_type  = Tensor::size_type;   // 大小 / 坐标类型

    // 构造：指定学习率 learning_rate（正数）。
    explicit SGD(value_type learning_rate);

    // 一步梯度下降：遍历网络所有 (参数, 梯度) 对，做 w -= lr * grad。
    //   实现思路：调用 net.parameters(visit)，在 visit 里对每对 (w, gw) 逐元素循环
    //     for (size_type i = 0; i < w.size(); ++i) w[i] -= lr * gw[i];
    void step(Network& net) const;

    // ---- 超参访问 ----
    value_type learning_rate() const noexcept { return learning_rate_; }  // 学习率

private:
    value_type learning_rate_;  // 学习率
};

} // namespace cnn
