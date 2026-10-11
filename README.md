# CNN — 手写数字识别（C++17 + CMake）

从零实现的一个小型卷积神经网络（CNN），在 [MNIST](https://en.wikipedia.org/wiki/MNIST_database) 手写数字数据集上完成训练与推理。核心层全部用 **C++17 纯标准库** 手写，不依赖 PyTorch / TensorFlow / Eigen 等第三方 ML 库。

- 完整支持 **前向传播 + 反向传播（自动手动求导）+ SGD 训练**
- 提供 **控制台训练程序**（打印每轮损失与准确率）
- 提供 **Qt6 可视化程序**（实时绘制损失 / 准确率曲线，展示样本预测）

---

## 特性

| 模块     | 头文件（相对 `include/cnn/`） | 说明                                                                   |
| -------- | ---------------------------- | ---------------------------------------------------------------------- |
| 张量     | `core/tensor.h`              | N 维张量，NCHW 布局、行主序 strides、`matmul`/`transpose`/`softmax` 等 |
| 卷积层   | `layers/conv2d.h`            | valid/same 卷积，Xavier 初始化，im2col 反向                            |
| 激活层   | `layers/relu.h`              | ReLU（另有 `layers/softmax.h` 独立层）                                 |
| 池化层   | `layers/pooling.h`           | MaxPool2D                                                              |
| 展平层   | `layers/flatten.h`           | `[N,C,H,W]` → `[N,C*H*W]`（纯 reshape）                                |
| 全连接层 | `layers/dense.h`             | `y = x @ W + b`，Xavier 初始化                                         |
| 网络容器 | `core/network.h`             | 类型擦除的顺序容器，统一驱动 forward/backward                          |
| 损失函数 | `training/loss.h`            | CrossEntropy（**自带数值稳定的 log-softmax**）                         |
| 优化器   | `training/optimizer.h`       | SGD（`w -= lr * grad`）                                                |
| 数据加载 | `data/dataset.h`             | Kaggle 格式 MNIST CSV 加载器                                           |
| 模型读写 | `io/model_io.h`              | 保存 / 加载网络参数                                                    |

> 注意：`CrossEntropy` 内部已包含 softmax，因此训练网络末尾**不要再接 `Softmax` 层**（否则会做两次 softmax）。

---

## 目录结构

```
CNN-MNIST/
├── CMakeLists.txt                 # 核心库与控制台构建配置，引入 GUI 子构建
├── CMakePresets.json              # 可共享的构建基础配置
├── CMakeUserPresets.json.template # 本机 Qt / MinGW / Ninja 配置参考
├── CMakeUserPresets.json          # 本机 Debug、Release 预设（Git 忽略）
├── .vscode/                       # CMake、IntelliSense、构建任务及 GDB 调试配置
├── include/cnn/                   # 核心库公共头文件
│   ├── core/                     # tensor.h、network.h
│   ├── layers/                   # conv2d / dense / flatten / pooling / relu / softmax
│   ├── training/                 # loss.h、optimizer.h
│   ├── data/                     # dataset.h
│   └── io/                       # model_io.h
├── src/                           # 实现文件，与公共头文件按相同模块分组
│   ├── core/
│   ├── layers/
│   ├── training/
│   ├── data/
│   ├── io/
│   ├── main.cpp                  # 前向传播演示（cnn_demo）
│   └── mnist_test.cpp            # 训练 / 评估 / 模型导出（mnist_test）
├── gui/                           # Qt6 可视化程序（mnist_gui）
│   ├── CMakeLists.txt             # GUI 子构建配置（由根目录 add_subdirectory 引入）
│   ├── include/                   # GUI 头文件，按职责分组
│   │   ├── windows/              # mainwindow.h、predictwindow.h：主窗口与预测窗口
│   │   ├── widgets/              # digitwidget.h、plotwidget.h：数字显示与曲线控件
│   │   └── training/             # trainer.h：后台训练工作对象
│   └── src/                       # 与 GUI 头文件按相同职责分组
│       ├── windows/
│       ├── widgets/
│       ├── training/
│       └── main.cpp              # Qt 应用入口
└── data/                          # 数据集（Kaggle CSV，见下）
```

查找一个模块时，先在 `include/cnn/<模块>/` 阅读接口，再到 `src/<模块>/`
查看同名实现；GUI 的 `include/` 与 `src/` 也采用相同分组。
头文件引用包含模块名，例如 `#include "cnn/layers/conv2d.h"`、
`#include "widgets/plotwidget.h"`。核心库与 GUI 使用递归源码收集，新增文件会
自动纳入源码列表；移动文件后重新运行 CMake 配置，以刷新 IntelliSense。

---

## 环境要求

- **Qt6 Widgets** 与配套的 **MinGW** 编译器
- **CMake** ≥ 3.21，需要能通过 `cmake` 命令调用
- **Ninja** 构建工具

本机路径集中在 `CMakeUserPresets.json`，该文件已加入 `.gitignore`。
CMake 和 VS Code CMake Tools 会自动读取它，并将预设中的环境变量传给配置、构建进程。

---

## 构建

### 1) 使用本机 Qt 预设

首次使用时复制参考模板，按实际安装位置修改 `environment` 中的
`QT_ROOT`、`MINGW_ROOT` 和 `NINJA_ROOT`，路径使用 `/`：

```bash
cp CMakeUserPresets.json.template CMakeUserPresets.json
```

模板会从这三个目录生成编译器、Qt、Ninja、GDB 和运行库路径。
已有本机配置时直接编辑 `CMakeUserPresets.json`，无需再次复制。

```bash
# Release：构建核心库、两个控制台程序和 Qt GUI
cmake --preset qt-mingw-release
cmake --build --preset qt-mingw-release --parallel

# Debug：供 GDB / VS Code 调试
cmake --preset qt-mingw-debug
cmake --build --preset qt-mingw-debug --parallel
```

输出分别位于 `build/qt-mingw-release/` 和 `build/qt-mingw-debug/`。
GUI 构建后会自动调用这套 Qt 的 `windeployqt`，将 Qt DLL、`platforms/qwindows.dll`
及 MinGW 运行库部署到程序旁；无需手动复制 DLL。
这套 Qt 只提供 Release 库，Debug 应用也使用这些库，部署工具会根据实际依赖判断。
如需自行管理运行库，可在配置时传入 `-DCNN_DEPLOY_QT=OFF`。

### 2) VS Code

安装 **CMake Tools** 和 **C/C++** 扩展，打开项目后通过
`CMake: Select Configure Preset` 选择 `qt-mingw-debug` 或 `qt-mingw-release`。
C/C++ 的头文件路径和宏由 CMake Tools 提供，配置成功后可识别 Qt 头文件和生成的 MOC 文件。

首次调试前先完成一次 `qt-mingw-debug` 配置；调试器和运行库目录会从 CMake 缓存读取。
按 `Ctrl+Shift+B` 构建 Debug；选择 `MNIST Qt GUI (Debug)` 后按 `F5`，
会先配置并构建，再用配套 GDB 启动 GUI。调试工作目录为项目根目录，以便读取 `data/`。

### 3) 仅构建控制台程序 / 其他机器

GUI 默认启用，Qt 未找到时会明确报错。无需 Qt 时可关闭 GUI：

```bash
cmake -S . -B build/console -G Ninja -DCMAKE_BUILD_TYPE=Release -DCNN_BUILD_GUI=OFF
cmake --build build/console --parallel
```

其他机器可复制模板建立自己的 `CMakeUserPresets.json`，或直接使用
`cmake -S . -B <新构建目录> -DCMAKE_PREFIX_PATH=<Qt目录> -DCMAKE_CXX_COMPILER=<匹配编译器>`。
切换编译器 / 生成器时应使用新的构建目录，避免旧 CMake 缓存冲突。

---

## 运行

**都在项目根目录下运行**（数据使用相对路径 `data/train.csv`）：

```powershell
# 控制台训练（默认：10000 训练 + 2000 验证，5 轮）
./build/qt-mingw-release/mnist_test.exe

# 前向传播演示
./build/qt-mingw-release/cnn_demo.exe

# 可视化程序
./build/qt-mingw-release/mnist_gui.exe
```

预设的 PATH 只作用于配置、构建进程，不会修改当前终端。
运行控制台程序时，如 PATH 中存在其他 MinGW，可从本机预设读取匹配的运行库目录：

```powershell
$localPreset = Get-Content CMakeUserPresets.json -Raw | ConvertFrom-Json
$toolPaths = ($localPreset.configurePresets | Where-Object name -eq 'qt-mingw-local').environment
$env:PATH = "$($toolPaths.MINGW_ROOT)/bin;$($toolPaths.QT_ROOT)/bin;$env:PATH"
```

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

| 文件        | 形状        | 说明                                          |
| ----------- | ----------- | --------------------------------------------- |
| `train.csv` | 42000 × 785 | 首列 `label`（0–9），其后 784 个像素（0–255） |
| `test.csv`  | 28000 × 784 | 仅 784 个像素，无标签                         |

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

约 36 秒 / 轮。可在 `src/mnist_test.cpp` 的匿名命名空间中调整 `kTrainSize` / `kValSize` / `kBatchSize` / `kEpochs` / `kLr` 等参数。

---

## 注意事项

- **数据文件读取使用 C 标准库**：旧 MSYS2 UCRT64 g++ 14.2 环境下曾出现 `std::ifstream` 优化后运行异常，`src/data/dataset.cpp` 使用 `fopen` / `fgets` / `strtof`。
- 训练样本/验证样本从 `train.csv` 前 `train_size + val_size` 行切分，未使用 `test.csv`（该文件无标签，用于提交预测）。
