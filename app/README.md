# 基于Lime算法的低光照度物流数量检测系统

使用方法

1. 下载`NCNN`加速库
首先下载`NCNN`库，并编译对应的库文件
```bash
git clone https://github.com/Tencent/ncnn.git
cd ncnn 
mkdir -p build 
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j$(nproc)
make install
```

然后将`ncnn/build/install`文件夹移动到本项目下面，并且将`install`文件夹命名为`ncnn`.

2. 下载`NCNN`模型文件
可以使用`NCNN`已经转化好的模型文件，下载地址为`https://github.com/shaoshengsong/yolov5_62_export_ncnn`。
本程序中使用到的文件是 `yolov5s_6.2.bin`和`yolov5s_6.2.param`。

在本项目下新建`models`文件夹，并将上述两个模型文件移动到`models`文件夹下面。

3. 编译项目
```bash
make all
```
4. 运行项目
```bash
cd build 
./uYoloAppLime
```
5. 清除项目
```bash
make clean
```
