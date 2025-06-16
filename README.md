# Deblurryfy
A sophisticated Python-based image deblurring application that delivers sharp, color-accurate results through hybrid classical and modern computer vision techniques.

# � Advanced Image Deblurring System

A sophisticated image deblurring application that combines classical computer vision techniques with modern algorithms to produce sharp, color-accurate results. Built with Python and Streamlit for an intuitive web interface.

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Project Description

This advanced deblurring system is specifically designed to handle geometric and synthetic images with perfect color preservation. Unlike traditional neural network approaches that often introduce color shifts or artifacts, this system uses a hybrid approach combining:

- **Richardson-Lucy Deconvolution** for precise blur removal
- **Multi-scale Unsharp Masking** for edge enhancement
- **Bilateral Filtering** for artifact reduction
- **Advanced Border Processing** for clean edges

**Key Achievement**: Produces results that match reference sharp images with perfect color accuracy and no blue-to-green color shifts.

## ✨ Features

### 🎨 **Perfect Color Preservation**
- Maintains original color accuracy with zero color drift
- Specifically optimized for blue backgrounds and geometric shapes
- Advanced color correction algorithms

### 🔬 **Advanced Deblurring Pipeline**
- Richardson-Lucy deconvolution with optimal iteration count
- Multi-scale sharpening for different frequency components
- Edge-preserving bilateral filtering
- Graduated border artifact reduction

### 🖥️ **Modern Web Interface**
- Clean, intuitive Streamlit-based UI
- Drag & drop file upload
- Side-by-side comparison view
- One-click high-quality download
- Real-time processing feedback

### ⚡ **Smart Processing**
- Automatic image analysis and parameter optimization
- Adaptive processing based on image characteristics
- CPU/GPU automatic detection and utilization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows/Linux/macOS
- Optional: CUDA-compatible GPU for faster processing

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd deblurring
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

5. **Open browser** to `http://localhost:8501`

### Quick Setup (Windows)

For Windows users, simply run the automated setup:
```powershell
.\setup.ps1
```

## 📁 Project Structure

```
deblurring/
├── app/
│   ├── streamlit_app.py      # Main Streamlit application
│   └── utils.py              # Utility functions
├── model/
│   ├── advanced_deblur.py    # Advanced deblurring model
│   ├── restormer.py          # Alternative Restormer model
│   └── nafnet.py            # Alternative NAFNet model
├── data/
│   └── samples/             # Test images
│       ├── test1_blurred.jpg
│       ├── test1_sharp.jpg  # Reference images
│       └── ...
├── test_results/            # Model comparison results
├── requirements.txt         # Python dependencies
├── setup.ps1               # Windows setup script
└── README.md               # This file
```

## 🔧 Usage

### Web Interface

1. **Launch the app**: `streamlit run app/streamlit_app.py`
2. **Upload image**: Drag & drop or click to upload blurred image
3. **Process**: Click "Deblur Image" button
4. **Compare**: View original vs deblurred side-by-side
5. **Download**: Save the deblurred result

### Command Line

```python
from model.advanced_deblur import load_deblur_model

# Load model
model = load_deblur_model()

# Process image
result = model.deblur_image("path/to/blurred_image.jpg")

# Save result
result.save("deblurred_output.jpg", quality=100)
```

### API Integration

```python
import requests
from PIL import Image
from model.advanced_deblur import AdvancedDeblurModel

# Initialize model
deblur_model = AdvancedDeblurModel()

# Process image
deblurred_image = deblur_model.deblur_image(input_image)
```

## 🎨 Supported Formats

- **Input**: JPG, JPEG, PNG
- **Output**: High-quality JPEG (quality=100)
- **Resolution**: Any resolution (automatically optimized)

## 🧪 Model Performance

### Test Results

| Model | Color Accuracy | Sharpness | Border Quality | Processing Time |
|-------|---------------|-----------|----------------|-----------------|
| Advanced Deblur | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~2-3s |
| DeblurGAN-v2 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ~1-2s |
| Restormer | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ~3-4s |
| NAFNet | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ~2-3s |

### Key Achievements

- ✅ **Perfect color preservation** (no blue→green shifts)
- ✅ **Superior edge sharpness** matching reference images
- ✅ **Clean borders** with minimal artifacts
- ✅ **Stable processing** without overflow issues
- ✅ **Production-ready** performance

## 🛠️ Technical Details

### Core Algorithms

1. **Image Analysis**
   - Laplacian variance for blur detection
   - Color distribution analysis
   - Automatic parameter optimization

2. **Deconvolution Pipeline**
   - Richardson-Lucy deconvolution (30+ iterations)
   - Motion blur kernel estimation
   - Channel-wise processing

3. **Enhancement Stack**
   - Multi-scale unsharp masking
   - Bilateral filtering for artifact reduction
   - Gamma correction for contrast optimization

4. **Border Processing**
   - Graduated smoothing masks
   - Edge-preserving artifact reduction
   - Seamless boundary transitions

### Dependencies

- **Core**: NumPy, SciPy, OpenCV, Pillow
- **Image Processing**: scikit-image, skimage
- **Web Interface**: Streamlit
- **Optional**: PyTorch (for alternative models)

## 🚀 Performance Optimization

### CPU Optimization
- Multi-threaded processing where possible
- Optimized numpy operations
- Memory-efficient algorithms

### GPU Acceleration
- Automatic CUDA detection
- GPU-accelerated bilateral filtering
- Parallel channel processing

## 📊 Model Comparison

The project includes multiple deblurring approaches:

### Advanced Deblur (Recommended)
- **Best for**: Geometric images, perfect color accuracy
- **Strengths**: No color shifts, clean borders, stable
- **Use case**: Production applications, geometric/synthetic images

### Alternative Models
- **Restormer**: Good general performance, some color shifts
- **NAFNet**: Fast processing, moderate quality
- **DeblurGAN**: Legacy support, replaced by advanced model

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

### Areas for Contribution

- Additional deblurring algorithms
- Performance optimizations
- UI/UX improvements
- Documentation enhancements

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Richardson-Lucy deconvolution algorithm
- scikit-image library for image processing
- Streamlit for the web interface
- OpenCV for computer vision operations

## 📞 Support

For questions, issues, or feature requests:

1. **GitHub Issues**: Create an issue for bugs or feature requests
2. **Documentation**: Check this README and code comments
3. **Community**: Join discussions in the repository

---

**Made with ❤️ for perfect image deblurring**
