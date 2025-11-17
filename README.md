<div align="center">

# 🎭 Deepfake Detection System

### AI-Powered Video Authenticity Analysis Using Computer Vision & Deep Learning

[![Version](https://img.shields.io/badge/version-1.0rc1-blue.svg)](https://github.com/comb0601/Deepfake_detection)
[![Status](https://img.shields.io/badge/status-rc-blue.svg)](https://github.com/comb0601/Deepfake_detection)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/comb0601/Deepfake_detection)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.3.0+-orange.svg)](https://www.tensorflow.org/)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-system-architecture) • [Performance](#-performance) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Feature Extraction](#-feature-extraction)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance](#-performance)
- [Requirements](#-requirements)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

A sophisticated deepfake detection system that leverages **computer vision features** and **deep neural networks** to identify manipulated videos. This system analyzes inter-frame variations in videos using advanced image processing techniques to distinguish between authentic and synthetic content.

### Key Highlights

- 🎯 **Multi-Feature Analysis**: Extracts 24+ computer vision features from video frames
- 🧠 **Deep Learning Models**: Implements MesoNet and custom CNN architectures
- 📊 **Frame-by-Frame Analysis**: Detects subtle temporal inconsistencies
- ⚡ **High Accuracy**: Achieves competitive performance on deepfake datasets
- 🔬 **Research-Grade**: Based on peer-reviewed deepfake detection methodologies

---

## 💡 Motivation

Deepfake technology uses artificial intelligence to synthesize and manipulate faces in videos, creating highly realistic but fake content. While this technology has legitimate applications, it poses significant risks:

- **Political Manipulation**: Creating false narratives and misinformation
- **Identity Theft**: Unauthorized use of personal likenesses
- **Malicious Content**: Non-consensual synthetic media
- **Erosion of Trust**: Undermining authenticity of digital media

This project aims to combat these threats by providing an accessible, open-source tool for deepfake detection.

---

## ✨ Features

### 🎨 Computer Vision Features (24 Features)

Our system extracts comprehensive inter-frame differences to capture temporal inconsistencies:

| Category | Features | Description |
|----------|----------|-------------|
| **Similarity Metrics** | MSE, PSNR, SSIM | Measures pixel-level and structural similarities |
| **Histogram Analysis** | Histogram Difference | Detects distribution changes in pixel intensities |
| **Color Space** | RGB, HSV | Analyzes color space variations (average & max values) |
| **Image Quality** | Luminance, Variance | Evaluates brightness and variance changes |
| **Edge Analysis** | Edge Density, Edge Entropy | Detects edge inconsistencies and noise |
| **Frequency Domain** | DCT Coefficients | Analyzes sharpness using Discrete Cosine Transform |
| **Statistical** | Entropy | Measures information content changes |

### 🧠 Deep Learning Models

- **Meso4**: 4-layer MesoNet architecture optimized for deepfake detection
- **MesoInception4**: Inception-based architecture with dilated convolutions
- **Custom DNN**: Variance-based classification on temporal features

---

## 🏗️ System Architecture

<div align="center">

![System Architecture](https://user-images.githubusercontent.com/55551567/118912037-25f9e600-b962-11eb-8498-be8c79b87422.png)

</div>

### Processing Pipeline

```
Video Input → Frame Extraction → Face Detection → Feature Extraction → Temporal Analysis → Classification → Deepfake Score
```

#### 1. **Preprocessing Phase**
   - Extract frames from video at specified intervals
   - Detect and crop faces using MTCNN (Multi-task Cascaded Convolutional Networks)
   - Normalize and resize images to 256×256

#### 2. **Feature Extraction Phase**
   - Compute 24 computer vision features between consecutive frames
   - Calculate inter-frame differences (MSE, PSNR, SSIM, etc.)
   - Extract color space statistics (RGB/HSV)
   - Analyze edge properties and frequency domain characteristics

#### 3. **Classification Phase**
   - Aggregate features over temporal windows
   - Calculate variance across frame sequences
   - Feed feature vectors into Deep Neural Network
   - Output probability score (0 = Real, 1 = Fake)

---

## 🔬 Feature Extraction

<div align="center">

![Features](https://user-images.githubusercontent.com/55551567/118912273-7f621500-b962-11eb-889b-ffda140ba2d4.png)

</div>

### Why These Features?

Deepfake generation processes introduce characteristic artifacts:

1. **Frame-to-Frame Synthesis**: Each frame is generated independently, causing temporal inconsistencies
2. **Resolution Limitations**: Target faces are extracted at limited resolution and scaled, reducing sharpness
3. **Compression Artifacts**: Format conversion introduces distortion and blur
4. **Color Space Anomalies**: Blending operations create unnatural color transitions
5. **Edge Inconsistencies**: Synthetic boundaries exhibit different properties than natural ones

### Temporal Variance Analysis

<div align="center">

![Variance Analysis](https://user-images.githubusercontent.com/55551567/118912104-3f029700-b962-11eb-9d6b-e67eb245bd4b.png)

</div>

The system identifies frames with significant rate of change, which often indicate manipulation points in deepfake videos.

---

## 📦 Installation

### Prerequisites

- Python 3.6 or higher
- CUDA-capable GPU (recommended for training)
- OpenCV 4

### Step 1: Clone the Repository

```bash
git clone https://github.com/comb0601/Deepfake_detection.git
cd Deepfake_detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow>=2.3.0
pip install keras>=2.4.3
pip install opencv-python>=4.0.0
pip install scikit-image>=0.17.2
pip install scikit-learn>=0.23.2
pip install numpy>=1.18.5
pip install pandas>=1.1.3
pip install facenet-pytorch
pip install matplotlib
pip install ipython>=7.18.1
```

---

## 🚀 Usage

### 1. Face Extraction from Video

```python
from face_extraction import extract_faces

# Extract faces from video
extract_faces('path/to/video.mp4', output_dir='faces/')
```

### 2. Feature Analysis

```python
from check import (
    get_image_difference,
    calcRGBCenter,
    totalEntropy,
    edgeDensityAnalysis
)

# Compute features between two frames
mse = np.sum((frame1.astype("float") - frame2.astype("float")) ** 2)
psnr = cv2.PSNR(frame1, frame2)
histogram_diff = get_image_difference(frame1, frame2)
rgb_diff = abs(calcRGBCenter(frame2) - calcRGBCenter(frame1))
```

### 3. Train Classifier

```python
from classifiers import Meso4, MesoInception4

# Initialize model
model = Meso4(learning_rate=0.001)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

### 4. Complete Analysis Pipeline

```bash
# Extract features from video dataset
python check.py --input dataset/videos/ --output results/

# Analyze features
python analyze.py --input results/ --output analysis.csv

# Train and evaluate classifier
python classifiers.py --train --data analysis.csv
```

---

## 📊 Performance

<div align="center">

![Performance Results](https://user-images.githubusercontent.com/55551567/118912468-cf40dc00-b962-11eb-83cd-363f6c198609.png)

</div>

### Evaluation Metrics

Our system demonstrates competitive performance across multiple deepfake detection benchmarks:

- **Accuracy**: High detection rates on FaceSwap and DeepFake datasets
- **Precision**: Low false positive rate
- **Recall**: Effective at identifying manipulated content
- **F1-Score**: Balanced performance across metrics

*Note: Detailed performance metrics depend on the specific dataset and training configuration.*

---

## 🛠️ Requirements

### Software Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.6+ | Core language |
| TensorFlow | 2.3.0+ | Deep learning framework |
| Keras | 2.4.3+ | Neural network API |
| OpenCV | 4.0+ | Computer vision operations |
| scikit-image | 0.17.2+ | Image processing |
| scikit-learn | 0.23.2+ | Machine learning utilities |
| NumPy | 1.18.5+ | Numerical computing |
| Pandas | 1.1.3+ | Data manipulation |
| facenet-pytorch | Latest | Face detection |
| IPython | 7.18.1+ | Interactive computing |

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU (slow training)
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 6GB+ VRAM
- **Storage**: 10GB+ for datasets and models

---

## 📁 Project Structure

```
Deepfake_detection/
│
├── 📄 README.md                    # Project documentation
├── 📓 deepfake.ipynb               # Main Jupyter notebook
├── 📓 deepfake_1_1_3.ipynb        # Alternative notebook version
│
├── 🐍 Python Scripts
│   ├── check.py                    # Feature extraction from videos
│   ├── analyze.py                  # Feature analysis utilities
│   ├── classifiers.py              # CNN model architectures
│   ├── face_extraction.py          # Face detection and cropping
│   ├── absdiff.py                  # Absolute difference computation
│   ├── mesonet.py                  # MesoNet implementation
│   ├── snd.py                      # Signal-to-noise utilities
│   └── standard.py                 # Standardization functions
│
├── 📊 Data
│   └── result.csv                  # Sample feature analysis results
│
└── 🖼️  Screenshots
    ├── Screenshot 2020-11-26 164626.png
    └── Screenshot 2020-11-26 164654.png
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork the repository**
   ```bash
   git fork https://github.com/comb0601/Deepfake_detection.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Open a Pull Request**

### Areas for Contribution

- 🎯 Improve detection accuracy
- 🚀 Optimize performance and speed
- 📚 Add support for new deepfake methods
- 🧪 Expand test coverage
- 📖 Improve documentation
- 🐛 Bug fixes

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

✅ Commercial use
✅ Modification
✅ Distribution
✅ Private use

⚠️ Liability
⚠️ Warranty

---

## 🙏 Acknowledgments

- **TensorFlow & Keras Teams** - Deep learning frameworks
- **OpenCV Community** - Computer vision library
- **FaceNet Authors** - Face detection model
- **MesoNet Paper** - Original deepfake detection architecture
- **Research Community** - Ongoing deepfake detection research

### References

- Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). MesoNet: a Compact Facial Video Forgery Detection Network
- Deepfake Detection Challenge (DFDC) Dataset
- FaceForensics++ Dataset

---

## 📞 Contact

**Project Maintainer**: [@comb0601](https://github.com/comb0601)

**Project Link**: [https://github.com/comb0601/Deepfake_detection](https://github.com/comb0601/Deepfake_detection)

---

<div align="center">

### ⭐ Star this repository if you find it useful!

**Built with ❤️ to combat synthetic media manipulation**

</div>
