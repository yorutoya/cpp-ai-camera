# C++ AI Camera

基于 **C++17、OpenCV DNN 和 YOLO11n ONNX** 的实时摄像头目标检测项目。程序显示检测框、类别、置信度，并统计当前画面中的人数和有人/无人状态。

## 功能

- 读取默认摄像头并实时检测 COCO 80 类目标。
- 显示目标类别、置信度和检测框。
- 使用 letterbox 保持输入比例，并将检测框还原到原始画面。
- 进行置信度过滤和按类别执行的非极大值抑制（NMS）。
- 统计目标总数、各类别数量和人数。
- 根据当前帧人数显示 `OCCUPIED` 或 `EMPTY`。
- 显示 FPS 和模型前向推理耗时。
- 支持按 `Q`、`Esc` 或关闭窗口退出。

## 项目结构

```text
cpp-ai-camera/
├── CMakeLists.txt
├── include/
│   ├── AnalysisResult.hpp
│   ├── CameraAnalyzer.hpp
│   ├── Detection.hpp
│   └── YoloDetector.hpp
├── src/
│   ├── CameraAnalyzer.cpp
│   ├── YoloDetector.cpp
│   └── main.cpp
├── models/
│   └── yolo11n.onnx       # 本地准备，不提交到 Git
├── .gitignore
├── LICENSE
└── README.md
```

处理流程：

```text
摄像头画面 → YoloDetector → 检测结果 → CameraAnalyzer → 统计结果 → 界面显示
```

`YoloDetector` 负责预处理、推理和检测结果解析；`CameraAnalyzer` 负责统计；`main.cpp` 负责摄像头读取和界面绘制。

## 环境要求

- 支持 C++17 的编译器。
- CMake 3.16 或更新版本。
- OpenCV 4 或更新版本，包含 core、imgproc、dnn、highgui 和 videoio 模块。
- 摄像头和可显示窗口的桌面环境。
- COCO YOLO11n 检测模型，ONNX 格式。

Windows 下应使用与 OpenCV 构建环境匹配的编译器，例如 MSYS2 UCRT64 GCC 搭配同环境的 OpenCV。Python 仅用于准备模型，C++ 程序运行时不需要激活 Python 虚拟环境。

## 获取代码

```powershell
git clone https://github.com/yorutoya/cpp-ai-camera.git
cd cpp-ai-camera
```

## 准备模型

如果已有可用的 `yolo11n.onnx`，直接放入项目根目录下的 `models` 文件夹。

也可以使用 Ultralytics 导出。下面在 Windows PowerShell 中创建独立环境，并通过环境中的 Python 运行命令：

```powershell
python -m venv yolo-env
.\yolo-env\Scripts\python.exe -m pip install ultralytics onnx
.\yolo-env\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640, batch=1, dynamic=False, opset=12)"
New-Item -ItemType Directory -Force models
Copy-Item yolo11n.onnx models/yolo11n.onnx
```

首次加载官方权重需要联网下载。导出过程可能安装额外依赖；不同 Ultralytics 与 OpenCV 版本的兼容性需实际验证。

当前解析器要求输入尺寸为 `640 × 640`，输出为原始 float32 检测张量 `[1,84,N]`：前 4 项为边框坐标，其余 80 项为类别得分。不要使用带内置 NMS、分割、姿态或自定义类别的模型直接替换。

模型导出参数说明见 [Ultralytics 官方文档](https://docs.ultralytics.com/modes/export/)。模型权重已被 `.gitignore` 排除。

## 编译与运行

以下示例使用 Windows、MSYS2 UCRT64 和 MinGW Makefiles。请先将 CMake 加入 PATH，并按实际安装位置修改 `OpenCV_DIR`；该目录必须包含 `OpenCVConfig.cmake`。

在项目根目录运行：

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
cmake -S . -B build -G "MinGW Makefiles" -DOpenCV_DIR=C:/msys64/ucrt64/lib/cmake/opencv5 -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
.\build\ai-camera.exe
```

不传参数时，程序从运行工作目录读取 `models/yolo11n.onnx`，因此上述命令应在项目根目录运行。也可以传入模型的绝对路径：

```powershell
.\build\ai-camera.exe "D:\models\yolo11n.onnx"
```

默认打开摄像头编号 `0`。

在 CLion 中使用匹配的工具链，并将运行配置的 Working directory 设置为项目根目录。如果更换生成器或工具链，请使用新的构建目录。

## 界面说明

| 字段 | 含义 |
| --- | --- |
| FPS | 当前循环的帧率估计 |
| Inference | 模型前向推理耗时，不包含全部预处理与绘制时间 |
| Objects | 当前帧保留的检测目标数 |
| People | 当前帧检测到的人数 |
| Status | 人数大于 0 时为 OCCUPIED，否则为 EMPTY |

这些统计是逐帧结果，不包含人员身份识别、跟踪或累计进出人数。遮挡、光照和低置信度可能导致计数或状态跳动。

## 常见问题

- **找不到模型**：检查 `models/yolo11n.onnx` 是否存在，以及运行工作目录是否为项目根目录；也可以通过命令行传入模型的绝对路径。
- **找不到 OpenCV**：将 `OpenCV_DIR` 指向包含 `OpenCVConfig.cmake` 的目录。
- **缺少 DLL 或链接失败**：检查编译器与 OpenCV 是否来自兼容工具链，OpenCV 的 DLL 目录是否已加入 PATH。
- **无法打开摄像头**：检查系统摄像头权限和设备占用情况；需要其他摄像头时修改 `cv::VideoCapture camera(0)` 的编号并重新编译。
- **模型加载或解析失败**：检查 OpenCV 与 ONNX 导出的兼容性，并确认模型符合上述输入输出格式。

## 许可证

本仓库附带 [MIT License](LICENSE)。Ultralytics、模型权重和其他依赖的许可信息请分别查看其官方说明；仓库中的 LICENSE 不替代第三方条款。
