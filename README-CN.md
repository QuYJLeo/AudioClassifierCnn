# 🎵 音频分类器 CNN - 基于卷积神经网络的音频分类应用

[![许可证](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![平台](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)
[![Kotlin](https://img.shields.io/badge/Kotlin-1.3+-orange.svg)](https://kotlinlang.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Latest-purple.svg)](https://onnxruntime.ai/)

## 📖 项目概述

**Audio Classifier CNN** 是一个功能强大的 Android 应用，它利用卷积神经网络（CNN）和 ONNX Runtime 在移动设备上直接对音频信号进行分类。本项目展示了如何在 Android 上将深度学习模型部署到生产环境中，实现实时推理能力。

### ✨ 核心特性

- 🎯 **高性能音频分类**：采用 CNN 架构实现精准的音频识别
- 📱 **端侧推理**：完全在 Android 设备上运行，无需云端连接
- 🔍 **MFCC 特征提取**：提取梅尔频率倒谱系数进行音频分析
- ⚡ **ONNX Runtime 集成**：利用微软的 ONNX Runtime 实现高效的模型执行
- 📦 **预训练模型**：包含开箱即用的 `best_model.onnx`，快速部署
- 🎨 **简洁的 Kotlin 架构**：采用 Kotlin 进行现代化 Android 开发

## 🏗️ 项目架构

```
AudioClassifierCnn/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── assets/              # 模型和测试音频文件
│   │   │   ├── java/com/audioclassifier/
│   │   │   │   ├── MainActivity.kt    # 主应用逻辑
│   │   │   │   ├── WavFile.kt         # WAV 文件处理
│   │   │   │   └── WavFileException.kt
│   │   │   └── res/                 # Android 资源文件
│   │   └── test/                   # 单元测试
│   └── build.gradle                # 应用配置
├── gradle/                        # Gradle 包装器
└── build.gradle                   # 项目配置
```

## 🚀 快速开始

### 环境要求

- 📱 Android Studio Arctic Fox 或更高版本
- ☕ JDK 11 或更高版本
- 📱 Android SDK (API 27+)
- 📱 Android 设备或模拟器 (API 27+)

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/yourusername/AudioClassifierCnn.git
   cd AudioClassifierCnn
   ```

2. **在 Android Studio 中打开**
   - 启动 Android Studio
   - 选择 "打开现有项目"
   - 导航到项目目录并打开

3. **同步 Gradle**
   - 等待 Android Studio 同步项目
   - 下载所有缺失的依赖

4. **运行应用**
   - 连接 Android 设备或启动模拟器
   - 点击 "运行" 按钮或按 `Shift + F10`

## 📊 工作原理

### 音频处理流程

1. **🎵 音频加载** → 加载 WAV 音频文件
2. **🔬 MFCC 提取** → 提取 13 个 MFCC 系数，使用 64 个滤波器
3. **🔧 特征处理** → 填充/截断到 128 帧，转换为 CHW 格式
4. **🧠 ONNX 推理** → 通过 CNN 模型运行
5. **📈 Softmax 计算** → 将对数转换为概率
6. **🏆 结果预测** → 获取概率最高的类别

### 技术详情

| 组件 | 详情 |
|------|------|
| **模型格式** | ONNX (`best_model.onnx`) |
| **输入形状** | [1, 13, 3, 128] - [批次, 通道, 高度, 宽度] |
| **MFCC 系数** | 13 个倒谱系数 |
| **采样率** | 16,000 Hz |
| **框架** | ONNX Runtime Android |
| **语言** | Kotlin |

## 💻 核心组件

### MainActivity.kt

主活动负责：
- ✅ ONNX 模型初始化
- ✅ WAV 文件读取和解析
- ✅ MFCC 特征提取
- ✅ 特征预处理和格式化
- ✅ 模型推理
- ✅ 结果解释

### 关键类

- **MainActivity**：协调整个分类流程
- **WavFile**：处理 WAV 音频文件的 I/O 操作
- **SpeechFeatures**：提供 MFCC 特征提取功能

## 🔧 配置说明

### 依赖项

项目使用以下核心库：

```gradle
// Android 版 ONNX Runtime
implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'

// 语音特征提取
implementation "com.github.MerlynMind:kotlin_speech_features:1.0.0"

// AndroidX 库
implementation 'androidx.core:core-ktx:1.3.0'
implementation 'androidx.appcompat:appcompat:1.1.0'
```

### 模型输入/输出

**输入张量：**
- 形状：`[1, 13, 3, 128]`
- 数据类型：Float32
- 格式：CHW（通道-高度-宽度）

**输出：**
- 通过 softmax 得到的类别概率
- 预测的类别索引

## 📁 资源文件

`assets/` 目录包含：

| 文件 | 描述 |
|------|------|
| `best_model.onnx` | 用于音频分类的预训练 CNN 模型 |
| `不同意.wav` | 测试音频样本 1 |
| `不愿意.wav` | 测试音频样本 2 |
| `不是.wav` | 测试音频样本 3 |
| `否.wav` | 测试音频样本 4 |
| `知道.wav` | 测试音频样本 5 |

## 🎯 使用示例

```kotlin
// 初始化 ONNX 模型
initOnnxModel()

// 加载和处理音频
val wav = loadWavFile(readAudioFileFromAsset("sample.wav"))
val floatArray = wav.map { it.toFloat() }.toFloatArray()

// 提取 MFCC 特征
val mfccFeatures = speechFeatures.mfcc(floatArray, 16000, nFilt = 64, numCep = 13)

// 运行推理
val processedFeatures = processFeatures(mfccFeatures)
val prediction = runOnnxInference(processedFeatures)

Log.d("结果", "预测类别: $prediction")
```

## 🔬 技术栈

| 层级 | 技术 |
|------|------|
| **UI** | Android XML 布局 |
| **语言** | Kotlin |
| **ML 运行时** | ONNX Runtime |
| **音频处理** | Kotlin 语音特征 |
| **构建系统** | Gradle |
| **最低 SDK** | API 27 (Android 8.1) |
| **目标 SDK** | API 28 (Android 9) |

## 🤝 贡献指南

欢迎贡献！以下是参与方式：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 🎓 微软 ONNX Runtime 团队提供出色的推理引擎
- 🎵 MerlynMind 提供 Kotlin 语音特征库
- 📱 Android 社区的持续支持

## 📞 联系方式

如有问题或反馈，请开启 Issue 或联系我们！

---

⭐ 如果您觉得这个项目有用，请给个 Star！
