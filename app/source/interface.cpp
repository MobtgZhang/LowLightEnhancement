#include "interface.hpp"
#include "ui_mainwindow.h"
#include "lime.hpp"
#include "yolov5.hpp"

# include <opencv2/opencv.hpp>
# include <QtGui/QGuiApplication>
# include <QtGui/QImage>
# include <QtGui/QScreen>
# include <QtWidgets/QFileDialog>
# include <QtWidgets/QMessageBox>
# include <QtCore/QFlags>
# include <iostream>
# include <QtGui/QPixmap>

# include <iostream>
# include <map>

MainWindow::MainWindow(QWidget *parent):QMainWindow(parent),ui(new Ui::MainWindow){
    ui->setupUi(this);
    this->loadInterface();
}

MainWindow::~MainWindow(){
    delete ui;
    delete this->cap;
    delete this->capTimer;
    delete this->detector;
    delete this->takeTimer;
}

void MainWindow::loadInterface(){
    this->window_title = QString("仓库物品检测系统");
    this->cap_time = MIN_TIME_LENGTH;
    this->load_bin_file = "yolov5s_6.2.bin";
    this->load_param_file = "yolov5s_6.2.param";
    this->width = 640;
    this->height = 480;
    this->during_time = time(NULL); //获取系统当前的时间
    this->setWindowTitle(this->window_title);
    // 设置摄像头
    this->cap = new cv::VideoCapture;
    this->cap->set(cv::CAP_PROP_FRAME_WIDTH, this->width);  //设置宽度
    this->cap->set(cv::CAP_PROP_FRAME_HEIGHT, this->height);  //设置长度
    // radio按键
    this->ui->radioButtonHand->setChecked(true);
    this->ui->radioButtonHand->setEnabled(false);
    this->ui->radioButtonAuto->setChecked(false);
    this->ui->radioButtonAuto->setEnabled(false);
    this->ui->plainTextEditTime->setPlainText(QString("%1").arg(this->cap_time));
    this->ui->plainTextEditTime->setEnabled(false);
    this->ui->pushButtonTake->setEnabled(false);
    this->ui->pushButtonSave->setEnabled(false);
    this->ui->limeCheckBox->setEnabled(false);
    QObject::connect(this->ui->pushButtonSave,&QPushButton::clicked,this,&MainWindow::onSavePicture);
    QObject::connect(this->ui->plainTextEditTime,&QPlainTextEdit::textChanged,this,&MainWindow::textTimeChange);
    QObject::connect(this->ui->radioButtonHand, &QRadioButton::toggled, this, &MainWindow::forbidRadioButtion);
    //pushButton按键
    // 设置信号槽
    QObject::connect(this->ui->pushButtonConnect, &QPushButton::clicked, this, &MainWindow::connectCamera);
    QObject::connect(this->ui->pushButtonExit, &QPushButton::clicked, this, &MainWindow::close);
    // 设置摄像头
    this->isConnect = false;
    this->capTimer = new QTimer(this);
    // 设置时间间隔
    // 设置定时器间隔为10毫秒
    this->capTimer->setInterval(10);
    // 设置信号槽
    QObject::connect(this->capTimer, &QTimer::timeout, this, &MainWindow::updateFrame);
    // 居中窗口
    this->center();
    // 设置YOLOV5检测器
    this->detector = new YoloV5Detector(this->load_param_file,this->load_bin_file);
    // 设置抓拍计时器
    this->takeTimer = new QTimer(this);
    this->takeTimer->setInterval(this->cap_time);
    QObject::connect(this->takeTimer,&QTimer::timeout,this,&MainWindow::displayAutoTake);
    // 手动抓拍按钮
    QObject::connect(this->ui->pushButtonTake,&QPushButton::clicked,this,&MainWindow::displayManualTake);
}
void MainWindow::forbidRadioButtion(){
    if(this->isConnect){
        this->ui->plainTextEditTime->setEnabled(!this->ui->plainTextEditTime->isEnabled());
    }else{
        this->ui->plainTextEditTime->setEnabled(false);
    }
}
// 设置窗口关闭事件
void MainWindow::closeEvent(QCloseEvent *event){
    this->quit();
    // 关闭窗口
    event->accept();
}

void MainWindow::quit(){
    this->isConnect = false;
    // 关闭摄像头
    this->capTimer->stop();
    this->cap->release();
}

void MainWindow::updateFrame(){
    time_t tmp_time = time(NULL);
    if(this->cap->isOpened()){
        *(this->cap) >> this->frame;
        cv::cvtColor(this->frame,this->frame,cv::COLOR_BGR2RGB);
        if (this->ui->limeCheckBox->isChecked()){
            double alpha = 2.0;
            double fai = 4.6;
            cv::Mat temp = frame.clone();
            temp.convertTo(temp, CV_32F, 1.0 / 255);
            multi_fusion(temp, this->frame_enhance, alpha, fai);
            this->frame_enhance.convertTo(this->frame_enhance, CV_8U, 255);
        }else{
            this->frame_enhance = this->frame.clone();
        }
        this->frame_recongnize = this->frame_enhance.clone();
        this->objects.clear();
        this->detector->detect_objects(this->frame,this->objects);
        this->detector->draw_objects(this->frame_recongnize,this->objects);
        QImage image = QImage((const unsigned char*)(this->frame.data), this->frame.cols, this->frame.rows, QImage::Format_RGB888);
        image = image.scaled(this->ui->screenSpotView->width(), this->ui->screenSpotView->height());
        QGraphicsScene* scene = new QGraphicsScene();
        scene->addPixmap(QPixmap::fromImage(image));
        this->ui->cameraView->setScene(scene);
    }else{
        QMessageBox::critical(nullptr,this->window_title, "摄像头未连接");
        this->isConnect = false;
        this->ui->pushButtonConnect->setText("连接");
        this->capTimer->stop();
        // 设置自动抓拍
        this->takeTimer->stop();
    }
}

void MainWindow::center(){
    //获取可用桌面大小
    QRect rect = QGuiApplication::screens().at(0)->geometry();
    //获取窗口大小
    QSize size = this->size();
    //移动窗口
    this->move((rect.width() - size.width()) / 2, (rect.height() - size.height()) / 2);
}


void MainWindow::textTimeChange(){
    bool isNumber = false;
    QString tmp_str = this->ui->plainTextEditTime->toPlainText();
    if (tmp_str.isEmpty()){
        this->cap_time = MIN_TIME_LENGTH;
        this->ui->plainTextEditTime->setPlainText(QString("%1").arg(2));
    }else{
        auto value = tmp_str.toInt(&isNumber);
        if (isNumber){
            this->cap_time = value;
        }else{
            this->cap_time = MIN_TIME_LENGTH;
            this->ui->plainTextEditTime->setPlainText(QString("%1").arg(this->cap_time));
        }
    }
    this->takeTimer->setInterval(this->cap_time);
}

void MainWindow::displayAutoTake(){
    if(this->ui->radioButtonAuto->isChecked()){
        QImage image = QImage((const unsigned char*)(this->frame_recongnize.data), this->frame_recongnize.cols, this->frame_recongnize.rows, QImage::Format_RGB888);
        image = image.scaled(this->ui->screenSpotView->width(), this->ui->screenSpotView->height());
        QGraphicsScene* scene = new QGraphicsScene();
        scene->addPixmap(QPixmap::fromImage(image));
        this->ui->screenSpotView->setScene(scene);
        this->writeList();
    }
}

void MainWindow::displayManualTake(){
    QImage image = QImage((const unsigned char*)(this->frame_recongnize.data), this->frame_recongnize.cols, this->frame_recongnize.rows, QImage::Format_RGB888);
    image = image.scaled(this->ui->screenSpotView->width(), this->ui->screenSpotView->height());
    QGraphicsScene* scene = new QGraphicsScene();
    scene->addPixmap(QPixmap::fromImage(image));
    this->ui->screenSpotView->setScene(scene);
    this->writeList();
}

void MainWindow::writeList(){
    this->ui->listWidget->clear();
    for(auto &obj:this->objects){
        if (this->map_objects.find(class_names[obj.label]) == this->map_objects.end()){
            this->map_objects[class_names[obj.label]] = 1;
        }
        this->map_objects[class_names[obj.label]] += 1;
    }
    // 显示到列表当中
    for(auto &item:this->map_objects){
        this->ui->listWidget->addItem(QString("对象\t%1\t的个数是\t%2\t").arg(QString::fromStdString(item.first)).arg(item.second));
    }
}


void MainWindow::connectCamera(){
    if (this->isConnect){
        this->isConnect = false;
        this->ui->pushButtonConnect->setText("连接");
        this->capTimer->stop();
        QGraphicsScene* scene = new QGraphicsScene();
        this->ui->cameraView->setScene(scene);
        this->ui->radioButtonHand->setEnabled(false);
        this->ui->radioButtonAuto->setEnabled(false);
        this->ui->plainTextEditTime->setEnabled(false);

        this->ui->pushButtonTake->setEnabled(false);
        this->ui->pushButtonSave->setEnabled(false);
        this->ui->limeCheckBox->setEnabled(false);
        // 设置自动抓拍
        this->takeTimer->stop();
    }else{
        this->isConnect = true;
        // 设置摄像头
        this->cap->open(0);
        this->ui->pushButtonConnect->setText("断开连接");
        this->capTimer->start(50);

        this->ui->radioButtonHand->setEnabled(true);
        this->ui->radioButtonAuto->setEnabled(true);
        if(this->ui->radioButtonAuto->isChecked()){
            this->ui->plainTextEditTime->setEnabled(true);
        }

        this->ui->pushButtonTake->setEnabled(true);
        this->ui->pushButtonSave->setEnabled(true);
        this->ui->limeCheckBox->setEnabled(true);
        // 设置自动抓拍
        this->takeTimer->start(30);
    }
}

void MainWindow::onSavePicture(){
    QFileDialog dialog(this, "保存文件");
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setDefaultSuffix("jpg");

    QFlags<QFileDialog::Option> options;
    options |= QFileDialog::DontUseNativeDialog;  // 禁用原生对话框
    dialog.setOptions(options);
    if (dialog.exec()) {
        auto filePath = dialog.selectedFiles().first();
        if (!filePath.isEmpty()) {
            cv::imwrite(filePath.toStdString(),this->frame_recongnize);
        }
    }
}

