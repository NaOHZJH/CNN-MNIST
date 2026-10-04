// src/model_io.cpp

#include "cnn/model_io.h"

#include <cstdint>   // std::uint32_t
#include <cstdio>    // std::fopen / std::fread / std::fwrite / std::fclose
#include <stdexcept> // std::runtime_error
#include <string>
#include <vector>

namespace cnn {

namespace {

constexpr std::uint32_t kMagic   = 0x434E4E50u; // "PNNC"（小端），用于识别模型文件
constexpr std::uint32_t kVersion = 1u;          // 文件格式版本号

// 写 / 读一个 u32（小端，原生字节序，单机使用足够）。
void put_u32(std::FILE* f, std::uint32_t v) {
    std::fwrite(&v, sizeof(v), 1, f);
}

std::uint32_t get_u32(std::FILE* f) {
    std::uint32_t v = 0;
    if (std::fread(&v, sizeof(v), 1, f) != 1)
        throw std::runtime_error("模型文件损坏：读取标量失败");
    return v;
}

// 把张量形状拼成 "[a, b, c]" 这样的字符串，用于报错信息。
std::string shape_to_string(const std::vector<Tensor::size_type>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

} // namespace

void save_model(const Network& net, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr)
        throw std::runtime_error("无法写入文件: " + path);

    // 先统计参数张量总数（每个 weight / bias 各计一个），用于写文件头。
    std::uint32_t count = 0;
    net.parameters([&](Tensor&, Tensor&) { ++count; });

    put_u32(f, kMagic);
    put_u32(f, kVersion);
    put_u32(f, count);

    // 按 parameters() 的固定遍历顺序逐个写出（参数, 梯度）对里的「参数」，
    // 梯度不写。顺序约定见 network.h：对每个含参层先 weight 再 bias。
    net.parameters([&](Tensor& w, Tensor&) {
        const auto& s = w.shape();
        put_u32(f, static_cast<std::uint32_t>(s.size()));
        for (Tensor::size_type d : s)
            put_u32(f, static_cast<std::uint32_t>(d));
        std::fwrite(w.data(), sizeof(float), w.size(), f);
    });

    std::fclose(f);
}

void load_model(Network& net, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr)
        throw std::runtime_error("无法打开文件: " + path);

    if (get_u32(f) != kMagic) {
        std::fclose(f);
        throw std::runtime_error("不是有效的模型文件（magic 不匹配）");
    }
    if (get_u32(f) != kVersion) {
        std::fclose(f);
        throw std::runtime_error("模型文件版本不受支持");
    }
    const std::uint32_t count = get_u32(f);

    // 按与保存时相同的遍历顺序，逐个读回张量并回填进参数。
    std::size_t idx = 0;
    net.parameters([&](Tensor& w, Tensor&) {
        if (idx >= count) {
            std::fclose(f);
            throw std::runtime_error("模型文件参数不足：网络参数多于文件");
        }

        const std::uint32_t ndim = get_u32(f);
        std::vector<Tensor::size_type> shape(ndim);
        for (auto& d : shape)
            d = get_u32(f);

        if (shape != w.shape()) {
            std::fclose(f);
            throw std::runtime_error("第 " + std::to_string(idx) +
                                     " 个参数形状不匹配：网络期望 " +
                                     shape_to_string(w.shape()) + "，文件为 " +
                                     shape_to_string(shape));
        }

        if (std::fread(w.data(), sizeof(float), w.size(), f) != w.size()) {
            std::fclose(f);
            throw std::runtime_error("模型文件数据不足：读取第 " +
                                     std::to_string(idx) + " 个参数失败");
        }
        ++idx;
    });

    std::fclose(f);

    if (idx != count)
        throw std::runtime_error("模型文件参数多于网络：文件有 " +
                                 std::to_string(count) + " 个，网络有 " +
                                 std::to_string(idx) + " 个");
}

} // namespace cnn
