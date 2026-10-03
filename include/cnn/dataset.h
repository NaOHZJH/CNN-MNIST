// include/cnn/dataset.h
// 数据集与 MNIST CSV 加载器（Kaggle 格式）。
//   train.csv：首列 label，其后 784 个像素（0..255），共 42000 行。
//   test.csv ：784 个像素（无 label），共 28000 行。
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cnn/tensor.h"

namespace cnn {

// 样本集：images 存平铺像素（每样本 784 个 float，值域 [0,1]），labels 存类别索引（0..9）。
struct Dataset {
    using size_type = Tensor::size_type;   // std::size_t

    std::vector<std::vector<float>> images;  // [N][784] 归一化像素
    std::vector<size_type> labels;           // [N] 类别索引（无标签集为空）

    size_type num_samples() const noexcept { return images.size(); }
    size_type num_features() const noexcept { return images.empty() ? 0 : images[0].size(); }
};

// 读取 Kaggle 格式的 MNIST CSV：
//   has_label = true  → train.csv（首列 label，其后 784 个像素）
//   has_label = false → test.csv （784 个像素，无 label）
//   max_rows > 0 时只读前 max_rows 行（0 表示全部）。
// 像素除以 255 归一化到 [0,1]。
Dataset load_mnist_csv(const std::string& csv_path, bool has_label, size_t max_rows = 0);

} // namespace cnn
