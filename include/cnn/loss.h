// include/cnn/loss.h
// 交叉熵损失（CrossEntropy）：用于多分类任务，内部自带数值稳定的 log-softmax。
//   输入 pred：shape [N, C]（网络输出的原始 logits，未经 softmax）
//   目标 target：shape [N] 的类别索引（整数 0..C-1）
//   输出：标量损失 = 对 batch 取平均的 -log(softmax(pred)[target])
//   - 数值稳定：softmax 前先减去每行最大值，避免 exp 上溢。
//   - 重要约定：本层自带 softmax，因此网络末尾【不要再接 Softmax 层】，
//     否则会做两次 softmax，导致结果错误。
//   - 反向传播：dL/dpred[n,c] = (softmax(pred)[n,c] - onehot(target[n])[c]) / N。
#pragma once

#include <vector>

#include "cnn/tensor.h"

namespace cnn {

class CrossEntropy {
public:
    using value_type = Tensor::value_type;  // 元素类型 float
    using size_type  = Tensor::size_type;   // 大小 / 坐标类型

    // 前向传播：计算平均损失。pred[N, C] 为 logits，target[N] 为类别索引。
    //   实现思路（数值稳定的 log-softmax + NLL）：
    //     对每个样本 n：
    //       m      = max_c pred[n,c]                        // 每行最大值，防 exp 溢出
    //       lse    = m + log( Σ_c exp(pred[n,c] - m) )      // log-sum-exp
    //       loss_n = lse - pred[n, target[n]]               // 等于 -log(softmax(pred)[target])
    //     返回 Σ_n loss_n / N。
    //   会校验 pred.dim()==2、target.size()==pred.shape()[0]、每个 target[n] < C，否则抛异常。
    value_type forward(const Tensor& pred, const std::vector<size_type>& target) const;

    // 反向传播：返回 dL/dpred，形状 [N, C]（已除以 N，即对 batch 取平均后的梯度）。
    //   实现思路：对每个样本 n 先算 softmax 概率 p_c = exp(pred[n,c]-m)/Σexp(·)，
    //   再令 dx[n,c] = (p_c - (c == target[n] ? 1 : 0)) / N。
    Tensor backward(const Tensor& pred, const std::vector<size_type>& target) const;

    // 本层无「关键参数」——无权重/偏置/梯度，纯函数式。
};

} // namespace cnn
