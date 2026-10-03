# CNN — 手写数字识别（C++17 + CMake）

从零实现的一个小型卷积神经网络（CNN），在 [MNIST](https://en.wikipedia.org/wiki/MNIST_database) 手写数字数据集上完成训练与推理。核心层全部用 **C++17 纯标准库** 手写，不依赖 PyTorch / TensorFlow / Eigen 等第三方 ML 库。

- 完整支持 **前向传播 + 反向传播（自动手动求导）+ SGD 训练**
- 提供 **控制台训练程序**（打印每轮损失与准确率）
- 提供 **Qt6 可视化程序**（实时绘制损失 / 准确率曲线，展示样本预测）

---

## 特性

| 模块 | 文件 | 说明 |
|------|------|------|
| 张量 | `include/cnn/tensor.h` | N 维张量，NCHW 布局、行主序 strides、`matmul`/`transpose`/`softmax` 等 |
| 卷积层 | `conv2d` | valid/same 卷积，Xavier 初始化，im2col 反向 |
| 激活层 | `relu` | ReLU（另有 `softmax` 独立层，见下） |
| 池化层 | `pooling` | MaxPool2D |
| 展平层 | `flatten` | `[N,C,H,W]` → `[N,C*H*W]`（纯 reshape） |
| 全连接层 | `dense` | `y = x @ W + b`，Xavier 初始化 |
| 网络容器 | `network` | 类型擦除的顺序容器，统一驱动 forward/backward |
| 损失函数 | `loss` | CrossEntropy（**自带数值稳定的 log-softmax**） |
| 优化器 | `optimizer` | SGD（`w -= lr * grad`） |
| 数据加载 | `dataset` | Kaggle 格式 MNIST CSV 加载器 |

> 注意：`CrossEntropy` 内部已包含 softmax，因此训练网络末尾**不要再接 `Softmax` 层**（否则会做两次 softmax）。

---

## 目录结构

```
CNN/
├── CMakeLists.txt          # 构建脚本（libcnn 静态库 + 多个可执行目标）
├── include/cnn/            # 头文件（层 / 张量 / 网络 / 损失 / 优化器 / 数据）
├── src/                    # 实现文件
│   ├── main.cpp            # 前向传播演示（cnn_demo）
│   └── mnist_test.cpp      # 控制台训练 / 测试程序（mnist_test）
├── gui/                    # Qt6 可视化程序（mnist_gui）
│   ├── main.cpp
│   ├── mainwindow.*        # 主窗口（参数面板 + 曲线图 + 样本预览）
│   ├── trainer.*           # 后台训练线程（QThread 工作线程）
│   ├── plotwidget.*        # QPainter 手绘折线图（不依赖 Qt Charts）
│   └── digitwidget.*       # 28×28 样本显示 + 预测/真实标签
└── data/                   # 数据集（Kaggle CSV，见下）
```

---

## 环境要求

- **编译器**：g++ 14.2（本机为 MSYS2 UCRT64，`C:\msys64\ucrt64\bin`）
- **CMake** ≥ 3.16（本机用 VS 2022 自带，或 MSYS2 内安装）
- **构建工具**：`mingw32-make`（MinGW Makefiles 生成器）
- **Qt6**（可选，仅构建可视化程序时需要）：Qt 6.12.0 mingw_64

---

## 构建

### 1) 控制台训练程序

```bash
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
mingw32-make -C build mnist_test
```

> 若 `cmake` / `mingw32-make` 不在 PATH，请使用完整路径，例如
> `C:\msys64\ucrt64\bin\mingw32-make.exe`。

### 2) Qt6 可视化程序（可选）

构建时指定 Qt 安装路径：

```bash
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=C:/Qt/6.12.0/mingw_64
mingw32-make -C build mnist_gui
```

> `find_package(Qt6 QUIET ...)` 未找到 Qt 时会自动跳过该目标，不影响其余构建。

---

## 运行

**都在项目根目录下运行**（数据使用相对路径 `data/train.csv`）：

```bash
# 控制台训练（默认：10000 训练 + 2000 验证，5 轮）
./build/mnist_test.exe

# 前向传播演示
./build/cnn_demo.exe

# 可视化程序
./build/mnist_gui.exe
```

`mnist_gui.exe` 若无法启动（提示缺少 Qt DLL），可任选其一：

1. 临时把 Qt bin 加入 PATH：`export PATH="/c/Qt/6.12.0/mingw_64/bin:$PATH"`；
2. 或用 `windeployqt` 把 Qt DLL 部署到 exe 旁，并补齐 MSYS2 运行库
   `libstdc++-6.dll`、`libgcc_s_seh-1.dll`、`libwinpthread-1.dll`（来自 `C:\msys64\ucrt64\bin`）。

---

## 网络结构

```
输入 [N, 1, 28, 28]
  → Conv2D(1, 8, 3, stride=1, pad=1)   [N, 8, 28, 28]
  → ReLU
  → MaxPool2D(2, 2)                     [N, 8, 14, 14]
  → Conv2D(8, 16, 3, stride=1, pad=1)   [N, 16, 14, 14]
  → ReLU
  → MaxPool2D(2, 2)                     [N, 16, 7, 7]
  → Flatten                             [N, 784]
  → Dense(784, 10)                      [N, 10]
  → CrossEntropy（自带 softmax）
```

权重用 Xavier(Glorot) 均匀分布初始化，优化器为 SGD。

---

## 数据格式

数据为 **Kaggle MNIST CSV**（`data/` 目录）：

| 文件 | 形状 | 说明 |
|------|------|------|
| `train.csv` | 42000 × 785 | 首列 `label`（0–9），其后 784 个像素（0–255） |
| `test.csv` | 28000 × 784 | 仅 784 个像素，无标签 |

加载器 `cnn::load_mnist_csv(path, has_label, max_rows)` 会把像素除以 255 归一化到 `[0, 1]`。

---

## 训练结果

默认配置（10000 训练样本 + 2000 验证样本，batch=32，lr=0.01，5 轮）：

```
epoch 1/5  loss = 1.6308  val_acc = 78.00%
epoch 2/5  loss = 0.4878  val_acc = 87.50%
epoch 3/5  loss = 0.3693  val_acc = 89.35%
epoch 4/5  loss = 0.3157  val_acc = 88.05%
epoch 5/5  loss = 0.2720  val_acc = 92.15%
```

约 36 秒 / 轮。可在 `src/mnist_test.cpp` 的 `main()` 顶部调整 `train_size` / `val_size` / `batch_size` / `epochs` / `lr` 等参数。

---

## 注意事项

- **文件读取请用 C 标准库，勿用 `std::ifstream`**：本机 g++ 14.2（MSYS2 UCRT64）在 `-O1` 及以上优化级别下会错误编译 `std::ifstream`，导致运行时段错误。`src/dataset.cpp` 已改用 `fopen` / `fgets` / `strtof` 绕开该问题。
- 训练样本/验证样本从 `train.csv` 前 `train_size + val_size` 行切分，未使用 `test.csv`（该文件无标签，用于提交预测）。
