// src/dataset.cpp

#include "cnn/dataset.h"

#include <cstdio>   // std::fopen / std::fgets
#include <cstdlib>  // std::strtof
#include <stdexcept>

namespace cnn {

namespace {

// 把一行逗号分隔的 CSV 解析成 float 序列。用 strtof 直接扫描，比 stringstream 快得多。
// line 必须是以 '\0' 结尾的字符串（fgets 保证这一点）。
void parse_csv_line(const char* line, std::vector<float>& out) {
    out.clear();
    const char* p = line;
    while (*p != '\0') {
        char* next = nullptr;
        float v = std::strtof(p, &next);
        out.push_back(v);
        if (next == p) { ++p; continue; }  // 防御：无法解析时前进一格，避免死循环
        p = next;
        while (*p == ',' || *p == '\r' || *p == '\n') ++p;  // 跳过分隔符
    }
}

} // namespace

Dataset load_mnist_csv(const std::string& csv_path, bool has_label, size_t max_rows) {
    // 注意：这里用 C 标准库 fopen/fgets，而不是 std::ifstream。
    // 原因：MSYS2 UCRT64 的 g++ 14.2.0 在 -O1 及以上优化级别下，会把 std::ifstream
    // 的内联构造函数错误编译，导致打开文件时直接段错误（只在非 main 函数中触发）。
    // 改用 C stdio 可完全绕开该工具链 bug，而且解析 CSV 更快。
    std::FILE* f = std::fopen(csv_path.c_str(), "rb");
    if (f == nullptr)
        throw std::runtime_error("无法打开文件: " + csv_path);

    Dataset ds;

    const size_t ncols = has_label ? 785 : 784;   // 每行应有的列数
    std::vector<float> vals;
    vals.reserve(ncols);

    // MNIST CSV 最长的一行是表头（pixel0..pixel783 等字段名，约 7KB），
    // 数据行更短（约 3KB），16KB 缓冲足够。fgets 会在行尾保留 '\r' / '\n'。
    char line[16384];

    // 跳过表头（第一行）
    if (std::fgets(line, sizeof(line), f) == nullptr) {
        std::fclose(f);
        return ds;
    }

    size_t rows = 0;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        if (line[0] == '\r' || line[0] == '\n' || line[0] == '\0') continue;  // 跳过空行
        parse_csv_line(line, vals);
        if (vals.size() < ncols) continue;        // 防御：跳过残缺行

        if (has_label)
            ds.labels.push_back(static_cast<Tensor::size_type>(vals[0]));   // 类别索引 0..9

        std::vector<float> img(784);
        const size_t off = has_label ? 1 : 0;
        for (size_t i = 0; i < 784; ++i)
            img[i] = vals[off + i] / 255.0f;      // 归一化到 [0,1]
        ds.images.push_back(std::move(img));

        if (max_rows > 0 && ++rows >= max_rows) break;
    }

    std::fclose(f);
    return ds;
}

} // namespace cnn
