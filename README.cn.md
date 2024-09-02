## 1. 概述

本文的详细介绍如何搭建运行本demo的环境以及如何使用本demo程序。

## 2. 安装python并配置python环境
下载并安装[Python Release Python 3.8.10](https://www.python.org/downloads/release/python-3810/)，并配置独立的python环境，相关指令如下：
```powershell
mkdir -p C:\Users\HCKTest\source\venvs
cd C:\Users\HCKTest\source\venvs
C:\Users\HCKTest\AppData\Local\Programs\Python\Python38\python.exe -m venv qai_3.8.10
& "C:\Users\HCKTest\source\venvs\qai_3.8.10\Scripts\Activate.ps1"
python.exe -m pip install --upgrade pip
```

## 3. 安装demo依赖的python软件模块
运行本demo需要安装numpy、torch、diffusers、accelerate、opencv、transformers、torchvision等软件包，相关执行如下：
```powershell
pip install wheel setuptools pybind11
pip install numpy torch diffusers accelerate opencv-python transformers torchvision
```

## 4. 安装QNN SDK（AI Engine Direct）
下载并安装[QNN SDK v2.23.0.240531](https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct)，并配置环境，相关指令如下：
```powershell
& "C:\Qualcomm\AIStack\QAIRT\2.23.0.240531\bin\envsetup.ps1"
```
并手动将C:\Qualcomm\AIStack\QAIRT\2.23.0.240531\lib\hexagon-v73\unsigned中的库文件libqnnhtpv73.cat和libQnnHtpV73Skel.so拷贝至目录C:\Qualcomm\AIStack\QAIRT\2.23.0.240531\lib\arm64x-windows-msvc中。另外，为了后续使用方便，可以执行notepad.exe $PROFILE命令打开powershell配置文档,并在其尾部添加下面两行：
```powershell
& "C:\Users\HCKTest\source\venvs\qai_3.8.10\Scripts\Activate.ps1"
& "C:\Qualcomm\AIStack\QAIRT\2.23.0.240531\bin\envsetup.ps1"
```

## 5. 下载源码并编译安装qai_appbuilder
下载[qai_appbuilder源码](https://github.com/quic/ai-engine-direct-helper)，编译生成qai_appbuilder的python安装包并安装，相关指令如下：
```powershell
mkdir -p C:\Users\HCKTest\source\repos
cd C:\Users\HCKTest\source\repos
git clone https://github.com/quic/ai-engine-direct-helper.git
cd C:\Users\HCKTest\source\repos\ai-engine-direct-helper\pybind
Remove-Item -Force pybind11
git clone https://github.com/pybind/pybind11.git
cd C:\Users\HCKTest\source\repos\ai-engine-direct-helper
python .\setup.py bdist_wheel
pip install dist\qai_appbuilder-2.24.0-cp38-cp38-qnn-2.23-win_amd64.whl
```

## 6. 运行本demo程序修复指定的图片
本demo程序源码aotgan.py中定义了两个全局变量QNN_SDK_PATH和IMAGE_FILE，其中QNN_SDK_PATH指向QNN SDK实际的安装路径，IMAGE_FILE指定了本demo程序需要修复的图片，您可以根据实际需要修改它们。运行本demo程序的指令：
```powershell
cd C:\Users\HCKTest\source\repos\QCS8380-AOTGAN-on-WoS
python ./aotgan.py
```
本demo程序使用鼠标左键绘制出图片中需要修复的区域，并辅以快捷键进行操作，其中快捷键说明如下：
- 空格键：修复处理
- r键：复位操作，以回退到最初状态
- k键：接受并保持当前的修复结果，并继续修复
- +键：画笔加粗1个单位，最粗48个单位，默认15个单位
- -键：画笔变细1个单位，最细 3个单位，默认15个单位
- c键：切换画笔模式，支持矩形框模式和自由模式，默认矩形框模式
- s键：保存当前修复结果，与源图片同目录
- Esc键：退出运行
