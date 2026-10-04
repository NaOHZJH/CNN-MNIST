// gui/predictwindow.h
// 独立的模型预测窗口：加载已训练好的模型参数（方案 A，见 include/cnn/model_io.h）
// 与 data/test.csv（无标签测试集），逐张显示图片与预测结果，供用户目视判断预测效果。
//   - 模型结构在此处按训练时完全一致地重建（方案 A 约定），再 load_model 回填参数。
//   - 测试集无标签，因此 DigitWidget 的「真实」列恒为「?」。
#pragma once

#include <QWidget>

#include <vector>

#include "cnn/network.h"

class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class DigitWidget;

class PredictWindow : public QWidget {
    Q_OBJECT
public:
    explicit PredictWindow(QWidget* parent = nullptr);

private slots:
    void loadModel();     // 加载模型文件并重建网络
    void loadData();      // 加载测试数据（data/test.csv）
    void prev();          // 上一张
    void next();          // 下一张
    void randomSample();  // 随机一张

private:
    void showSample(int idx);  // 显示并预测第 idx 个样本（0 基，自动夹取到合法范围）
    void updateUiState();      // 根据数据/模型是否就绪来启用导航控件

    cnn::Network net_;                         // 已加载的模型
    bool modelLoaded_ = false;                 // 模型是否已加载成功
    std::vector<std::vector<float>> images_;   // 测试图片 [N][784]（已归一化）
    bool dataLoaded_ = false;                  // 数据是否已加载成功
    int  curIdx_ = 0;                          // 当前样本下标

    QLineEdit*   modelPathEdit_ = nullptr;
    QLineEdit*   dataPathEdit_  = nullptr;
    QPushButton* loadModelBtn_  = nullptr;
    QPushButton* loadDataBtn_   = nullptr;
    QSpinBox*    indexSpin_     = nullptr;
    QPushButton* prevBtn_       = nullptr;
    QPushButton* nextBtn_       = nullptr;
    QPushButton* randomBtn_     = nullptr;
    QLabel*      statusLabel_   = nullptr;
    DigitWidget* digit_         = nullptr;
};
