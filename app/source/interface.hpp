#ifndef YOLOAPPWINDOW_H
#define YOLOAPPWINDOW_H

# include "yolov5.hpp"
#include <QtWidgets/QMainWindow>
#include <QtCore/QString>
#include <QtCore/QTimer>
#include <QtGui/QCloseEvent>
#include <opencv2/opencv.hpp>
#include <ctime>
# include <string>
# define MIN_TIME_LENGTH 200

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
private:
    std::vector<Object> objects;
    std::map<std::string,unsigned int> map_objects;
    YoloV5Detector *detector;
    std::string load_param_file;
    std::string load_bin_file;
    Ui::MainWindow *ui;
    // 设置界面的初始化参数
    void loadInterface();
    // 设置标题
    QString window_title;
    //设置计时器，用于连接摄像头
    QTimer *capTimer;
    //设置计时器，用于抓拍
    QTimer *takeTimer;
    //判断是否连接
    bool isConnect;
    // 抓拍间隔时间
    unsigned int cap_time;
    // 设置摄像头
    cv::VideoCapture* cap;
    // 将窗口设置到屏幕的中央
    void center();
    // 设置窗口关闭事件
    void closeEvent(QCloseEvent *event);
    // 设置关闭事件
    void quit();
    // 设置计时器
    time_t during_time;
    // 设置抓拍帧
    cv::Mat frame;
    // 设置识别到的图片
    cv::Mat frame_recongnize;
    // 设置增强光照后的图片
    cv::Mat frame_enhance;
    // 设置的长和宽度
    unsigned int width;
    unsigned int height;
    // 将对象写在列表当中
    void writeList();
private slots:
    // 设置摄像头
    void updateFrame();
    // 设置摄像头连接
    void connectCamera();
    // 设置自动抓拍还是手动抓拍
    void forbidRadioButtion();
    // 将抓拍的图片显示到界面上
    void displayAutoTake();
    void displayManualTake();
    // 文本框改变的事件
    void textTimeChange();
    // 将图片保存下来
    void onSavePicture();
};

#endif // YOLOAPPWINDOW_H
