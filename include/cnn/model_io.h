// include/cnn/model_io.h
// 模型参数的导出 / 恢复（方案 A：只序列化权重与偏置，不含网络结构）。
//   - 保存：把网络里所有含参层（Conv2D / Dense）的 weight、bias 按层顺序写入二进制文件。
//   - 恢复：按相同顺序读回，逐个张量校验形状后回填到【已按相同结构构建好】的网络里。
//   关键约定：加载前必须在代码里把网络按训练时一模一样地重建，再调用 load_model。
//   因 Network 是类型擦除的（见 network.h），它无法描述自身结构，所以本模块不负责重建网络。
//   - 文件读写用 C 标准库（fopen/fread/fwrite），而非 std::ifstream——原因见 dataset.cpp：
//     本机 g++ 14.2（MSYS2 UCRT64）在 -O1 及以上会把 std::ifstream 错误编译导致段错误。
#pragma once

#include <string>

#include "cnn/network.h"

namespace cnn {

// 把网络所有可学习参数按层顺序写入二进制文件 path。
//   文件格式（小端）：
//     [magic:  u32]  魔数，用于识别文件
//     [version: u32] 格式版本号
//     [count:  u32]  参数张量总数（每个 weight / bias 各计一个）
//     对每个张量：
//       [ndim: u32]           维数
//       [shape[0..ndim-1]]    各维大小（每个 u32）
//       [data: size 个 float] 原始字节，按行主序
//   只保存参数，不保存梯度（梯度是训练态，恢复时无需）。
void save_model(const Network& net, const std::string& path);

// 从 save_model 生成的文件读回参数，按相同顺序回填进 net。
//   - net 必须是「已按相同结构构建好」的网络（层类型、超参、顺序一致）。
//   - 会校验 magic / version，并逐张量校验形状；张量个数或形状不一致时抛 std::runtime_error。
void load_model(Network& net, const std::string& path);

} // namespace cnn
