// gui/mainwindow.h
// 主窗口：训练参数设置 + 损失/准确率曲线 + 样本预测预览，驱动后台训练线程。
#pragma once

#include <QMainWindow>

class QLabel;
class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QLineEdit;
class PlotWidget;
class DigitWidget;
class Trainer;
class QThread;
class PredictWindow;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void startTraining();
    void stopTraining();
    void onEpochFinished(int epoch, int total, double loss, double valAcc,
                         double seconds, int pred, int truth);
    void onLogMessage(const QString& msg);
    void onTrainingFinished();
    void openPredict();

private:
    void buildUi();

    // 控件
    QSpinBox* epochsSpin_ = nullptr;
    QSpinBox* batchSpin_ = nullptr;
    QSpinBox* trainSpin_ = nullptr;
    QSpinBox* valSpin_ = nullptr;
    QDoubleSpinBox* lrSpin_ = nullptr;
    QPushButton* startBtn_ = nullptr;
    QPushButton* stopBtn_ = nullptr;
    QLineEdit* modelPathEdit_ = nullptr;
    QPushButton* predictBtn_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    PlotWidget* lossPlot_ = nullptr;
    PlotWidget* accPlot_ = nullptr;
    DigitWidget* digit_ = nullptr;

    // 训练线程（每次「开始」都新建一对，训练结束自动销毁）
    QThread* thread_ = nullptr;
    Trainer* trainer_ = nullptr;

    // 独立的预测窗口（按需创建，关闭后自动销毁）
    PredictWindow* predictWin_ = nullptr;
};
