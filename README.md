# 🎵 Audio Classifier CNN - 基于卷积神经网络的音频分类器

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)
[![Kotlin](https://img.shields.io/badge/Kotlin-1.3+-orange.svg)](https://kotlinlang.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Latest-purple.svg)](https://onnxruntime.ai/)

## 📖 Overview

**Audio Classifier CNN** is a powerful Android application that leverages Convolutional Neural Networks (CNN) and ONNX Runtime to classify audio signals directly on mobile devices. This project demonstrates how to deploy deep learning models in production on Android with real-time inference capabilities.

### ✨ Key Features

- 🎯 **High-performance Audio Classification**: Uses CNN architecture for accurate audio recognition
- 📱 **On-device Inference**: Runs entirely on Android devices without requiring cloud connectivity
- 🔍 **MFCC Feature Extraction**: Extracts Mel-Frequency Cepstral Coefficients for audio analysis
- ⚡ **ONNX Runtime Integration**: Leverages Microsoft's ONNX Runtime for efficient model execution
- 📦 **Pre-trained Model**: Includes a ready-to-use `best_model.onnx` for quick deployment
- 🎨 **Clean Kotlin Architecture**: Modern Android development with Kotlin

## 🏗️ Architecture

```
AudioClassifierCnn/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── assets/              # Model and test audio files
│   │   │   ├── java/com/audioclassifier/
│   │   │   │   ├── MainActivity.kt    # Main application logic
│   │   │   │   ├── WavFile.kt         # WAV file handling
│   │   │   │   └── WavFileException.kt
│   │   │   └── res/                 # Android resources
│   │   └── test/                   # Unit tests
│   └── build.gradle                # App configuration
├── gradle/                        # Gradle wrapper
└── build.gradle                   # Project configuration
```

## 🚀 Getting Started

### Prerequisites

- 📱 Android Studio Arctic Fox or later
- ☕ JDK 11 or higher
- 📱 Android SDK (API 27+)
- 📱 Android device or emulator with API 27+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AudioClassifierCnn.git
   cd AudioClassifierCnn
   ```

2. **Open in Android Studio**
   - Launch Android Studio
   - Select "Open an existing project"
   - Navigate to the project directory and open it

3. **Sync Gradle**
   - Wait for Android Studio to sync the project
   - Download any missing dependencies

4. **Run the application**
   - Connect your Android device or start an emulator
   - Click the "Run" button or press `Shift + F10`

## 📊 How It Works

### Audio Processing Pipeline

1. **🎵 Audio Loading** → Load WAV audio file
2. **🔬 MFCC Extraction** → Extract 13 MFCC coefficients with 64 filters
3. **🔧 Feature Processing** → Pad/truncate to 128 frames, convert to CHW format
4. **🧠 ONNX Inference** → Run through CNN model
5. **📈 Softmax Calculation** → Convert logits to probabilities
6. **🏆 Result Prediction** → Get class with highest probability

### Technical Details

| Component | Details |
|-----------|---------|
| **Model Format** | ONNX (`best_model.onnx`) |
| **Input Shape** | [1, 13, 3, 128] - [Batch, Channels, Height, Width] |
| **MFCC Coefficients** | 13 cepstral coefficients |
| **Sample Rate** | 16,000 Hz |
| **Framework** | ONNX Runtime Android |
| **Language** | Kotlin |

## 💻 Core Components

### MainActivity.kt

The main activity handles:
- ✅ ONNX model initialization
- ✅ WAV file reading and parsing
- ✅ MFCC feature extraction
- ✅ Feature preprocessing and formatting
- ✅ Model inference
- ✅ Result interpretation

### Key Classes

- **MainActivity**: Orchestrates the entire classification pipeline
- **WavFile**: Handles WAV audio file I/O operations
- **SpeechFeatures**: Provides MFCC feature extraction capabilities

## 🔧 Configuration

### Dependencies

The project uses these key libraries:

```gradle
// ONNX Runtime for Android
implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'

// Speech feature extraction
implementation "com.github.MerlynMind:kotlin_speech_features:1.0.0"

// AndroidX libraries
implementation 'androidx.core:core-ktx:1.3.0'
implementation 'androidx.appcompat:appcompat:1.1.0'
```

### Model Input/Output

**Input Tensor:**
- Shape: `[1, 13, 3, 128]`
- Data type: Float32
- Format: CHW (Channel-Height-Width)

**Output:**
- Class probabilities via softmax
- Predicted class index

## 📁 Assets

The `assets/` directory contains:

| File | Description |
|------|-------------|
| `best_model.onnx` | Pre-trained CNN model for audio classification |
| `不同意.wav` | Test audio sample 1 |
| `不愿意.wav` | Test audio sample 2 |
| `不是.wav` | Test audio sample 3 |
| `否.wav` | Test audio sample 4 |
| `知道.wav` | Test audio sample 5 |

## 🎯 Usage Example

```kotlin
// Initialize ONNX model
initOnnxModel()

// Load and process audio
val wav = loadWavFile(readAudioFileFromAsset("sample.wav"))
val floatArray = wav.map { it.toFloat() }.toFloatArray()

// Extract MFCC features
val mfccFeatures = speechFeatures.mfcc(floatArray, 16000, nFilt = 64, numCep = 13)

// Run inference
val processedFeatures = processFeatures(mfccFeatures)
val prediction = runOnnxInference(processedFeatures)

Log.d("Result", "Predicted class: $prediction")
```

## 🔬 Technical Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Android XML Layouts |
| **Language** | Kotlin |
| **ML Runtime** | ONNX Runtime |
| **Audio Processing** | Kotlin Speech Features |
| **Build System** | Gradle |
| **Min SDK** | API 27 (Android 8.1) |
| **Target SDK** | API 28 (Android 9) |

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🎓 Microsoft ONNX Runtime team for the excellent inference engine
- 🎵 MerlynMind for the Kotlin speech features library
- 📱 Android community for continuous support

## 📞 Contact

For questions or feedback, please open an issue or reach out!

---

⭐ Star this repo if you find it useful!
