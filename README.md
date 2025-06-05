# 🎽 Virtual Shirt Try-On System

<div align="center">

![Virtual Try-On Demo](https://img.shields.io/badge/Status-Fully_Working-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-yellow)

**Advanced AI-powered virtual try-on system using computer vision and pose estimation**

[🚀 Quick Start](#-quick-start) • [📋 Features](#-features) • [🛠️ Installation](#️-installation) • [📱 Usage](#-usage) • [🔧 Troubleshooting](#-troubleshooting)

</div>

---

## 📋 Features

### 🎯 **Three Powerful Implementations**
- **🆕 Simple Perfect Try-On** - Reliable, easy-to-use version with background removal
- **🔬 Ultimate Precision** - Research-grade algorithms with advanced warping
- **🎯 Super Accurate Fit** - Maximum precision using all 33 pose landmarks

### 🌟 **Core Capabilities**
- **✅ Real-time pose detection** using MediaPipe
- **🎭 Advanced background removal** with selfie segmentation
- **👕 Smart garment fitting** based on body measurements
- **🎨 High-quality alpha blending** for realistic overlay
- **📸 Screenshot capture** with organized storage
- **🔄 Live garment switching** during runtime
- **🪞 Mirror mode** for natural interaction

### 🔧 **Technical Features**
- **LANCZOS4 interpolation** for highest quality resizing
- **Perspective correction** for 3D realism
- **Temporal smoothing** for stable tracking
- **Multi-format support** (PNG, JPG, JPEG)
- **Automatic garment detection** from multiple folders
- **Edge smoothing** for natural blending

---

## 🛠️ Installation

### 📋 Prerequisites
- **Python 3.8+** 
- **Webcam** or camera device
- **macOS/Windows/Linux** supported

### 🔧 Required Dependencies

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### 📁 Project Structure
```
Virtual-Shirt-Try-On/
├── 📄 README.md                    # This documentation
├── 📄 FIXES_SUMMARY.md             # Technical fixes applied
├── 📄 requirements.txt             # Python dependencies
├── 🐍 simple_perfect_tryon_2.py    # Main simple version
├── 🐍 actually_working_fixed_tryon.py  # Reliable implementation
├── 🐍 ultimate_precision_tryon.py  # Research-grade version  
├── 🐍 super_accurate_perfect_fit.py # Maximum accuracy version
├── 🐍 test_all_fixes.py           # System verification
├── 📁 Garments/
│   ├── 📁 tops/                   # Shirt/top garments
│   │   ├── 🖼️ 1.png
│   │   ├── 🖼️ 3.png
│   │   └── 🖼️ 4.png
│   └── 📁 bottoms/                # Future: pants/skirts
├── 📁 Shirts/                     # Additional garments
│   ├── 🖼️ 1.png
│   ├── 🖼️ 3.png
│   └── 🖼️ 4.png
├── 📁 Screenshots/                # Saved screenshots
└── 📁 different test versions/    # Development versions
```

---

## 🚀 Quick Start

### 1️⃣ **Verify Installation**
```bash
python test_all_fixes.py
```
Expected output:
```
✅ Actually Working Fixed Try-On: PASSED
✅ Ultimate Precision Try-On: PASSED  
✅ Super Accurate Perfect Fit: PASSED
📊 Summary: 3/3 tests passed
🎉 ALL FIXES WORKING CORRECTLY!
```

### 2️⃣ **Choose Your Version**

#### 🌟 **Recommended: Simple Perfect Try-On**
```bash
python simple_perfect_tryon_2.py
```
- **Best for**: Beginners, reliable everyday use
- **Features**: Background removal, pose detection, smooth garment fitting

#### 🆕 **Alternative: Simple Working Try-On**
```bash
python simple_perfect_tryon_1.py
```
- **Best for**: Guaranteed reliability, proven working solution
- **Features**: Built on tested foundation, no complex algorithms

---

## 📱 Usage

### 🎮 **Controls (All Versions)**
| Key | Action | Description |
|-----|--------|-------------|
| **N** | Next Garment | Switch to next available garment |
| **P** | Previous Garment | Switch to previous garment |
| **S** | Save Screenshot | Capture current view to Screenshots/ |
| **ESC** | Exit | Close the application |

### 📸 **Screenshot Management**
- **Location**: All screenshots saved to `Screenshots/` folder
- **Format**: High-quality JPEG files
- **Naming**: `{version}_{timestamp}.jpg`
- **Examples**:
  - `Screenshots/perfect_tryon_1749108693.jpg`
  - `Screenshots/ultimate_precision_1749108734.jpg`
  - `Screenshots/super_accurate_fit_1749108801.jpg`

### 👕 **Adding New Garments**

1. **Prepare garment images**:
   - **Format**: PNG (with transparency) or JPG
   - **Background**: White or transparent preferred
   - **Size**: Any size (auto-resized)
   - **Quality**: Higher resolution = better results

2. **Add to folders**:
   ```bash
   # For tops/shirts
   cp your_garment.png Garments/tops/
   
   # Or to Shirts folder
   cp your_garment.png Shirts/
   ```

3. **Restart application** - new garments detected automatically

### 🎯 **Optimal Usage Tips**

#### 📷 **Camera Setup**
- **Distance**: Stand 6-8 feet from camera
- **Lighting**: Even, natural lighting preferred
- **Background**: Plain, contrasting background
- **Position**: Face camera directly, arms visible

#### 🧍 **Body Positioning**
- **Pose**: Stand upright, shoulders square to camera
- **Arms**: Slightly away from body for best detection
- **Visibility**: Ensure full torso is visible
- **Movement**: Keep relatively still for stable tracking

#### 👔 **Garment Selection**
- **PNG files**: Better for transparent backgrounds
- **High contrast**: Garments that contrast with your clothing
- **Clean edges**: Well-cut garment images work best

---

## 🔧 Technical Details

### 🤖 **AI Models Used**
- **MediaPipe Pose**: 33-landmark full-body detection
- **MediaPipe Selfie Segmentation**: Advanced background removal  
- **OpenCV**: High-quality image processing

### 🎨 **Image Processing Pipeline**
1. **Capture**: Real-time video from webcam
2. **Detection**: Pose landmarks + background segmentation
3. **Measurement**: Body proportions calculation
4. **Fitting**: Smart garment sizing and positioning
5. **Blending**: Alpha compositing with edge smoothing
6. **Display**: Real-time preview with UI overlay

### ⚙️ **Performance Optimizations**
- **LANCZOS4**: Highest quality image interpolation
- **Temporal smoothing**: Reduces jitter in tracking
- **Efficient memory management**: Optimized for real-time
- **Adaptive sizing**: Smart garment scaling algorithms

### 🔧 **Configuration Options**

#### Camera Settings (customizable in code):
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Height
```

#### MediaPipe Settings:
```python
# Pose detection accuracy
model_complexity=1          # 0=Light, 1=Full, 2=Heavy
min_detection_confidence=0.5 # 0.0-1.0
min_tracking_confidence=0.5  # 0.0-1.0
```

---

## 🔧 Troubleshooting

### ❌ **Common Issues**

#### 🚫 **"Cannot access camera"**
```bash
# Check camera permissions
# macOS: System Preferences > Security & Privacy > Camera
# Restart terminal/IDE after granting permissions
```

#### 🚫 **"No garments found"**
```bash
# Verify garment folders exist and contain images
ls Garments/tops/
ls Shirts/

# Add sample garments
cp sample_shirt.png Garments/tops/
```

#### 🚫 **"Module not found" errors**
```bash
# Install missing dependencies
pip install opencv-python mediapipe numpy

# Verify installation
python -c "import cv2, mediapipe, numpy; print('All modules OK')"
```

#### 🚫 **Poor pose detection**
- **Lighting**: Improve room lighting
- **Background**: Use plain, contrasting background
- **Distance**: Adjust distance from camera
- **Clothing**: Wear fitted clothing for better detection

#### 🚫 **Garment appears upside-down**
- **✅ Fixed in all versions!** - This was a known issue that has been resolved
- If still occurs, use `simple_perfect_tryon_2.py` (most reliable)

### 🛠️ **Performance Issues**

#### 🐌 **Slow performance**
```python
# Reduce camera resolution in code
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Lower MediaPipe complexity
model_complexity=0  # Instead of 1 or 2
```

#### 💾 **High memory usage**
- Close other applications
- Use `simple_perfect_tryon_2.py` for lowest memory usage
- Restart application periodically for long sessions

### 🔍 **Debug Mode**

To enable detailed logging, modify any version:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📈 Recent Fixes & Updates

### ✅ **Version 2.0 - All Issues Resolved!**

#### 🔧 **Fixed Issues**:
1. **OpenCV Parameter Error** - `ultimate_precision_tryon.py`
   - ❌ `interpolation=cv2.INTER_CUBIC` 
   - ✅ `flags=cv2.INTER_CUBIC`

2. **Upside-Down Garments** - `super_accurate_perfect_fit.py`
   - ❌ Unlimited rotation causing 180° flips
   - ✅ Conservative rotation limits (±15°)

3. **Screenshot Organization** - All versions
   - ❌ Screenshots saved to root folder
   - ✅ Screenshots organized in `Screenshots/` folder

#### 🆕 **New Features**:
- **Automatic folder creation** for screenshots
- **Improved error handling** across all versions
- **Comprehensive test suite** (`test_all_fixes.py`)
- **Better documentation** with this README

---

## 🤝 Contributing

### 🔄 **Development Workflow**
1. **Test changes** with `python test_all_fixes.py`
2. **Verify all versions** work correctly
3. **Update documentation** if needed
4. **Check screenshot saving** works properly

### 🧪 **Adding New Features**
- Follow existing code structure
- Test across all three main versions
- Ensure backward compatibility
- Update this README with new features

### 🐛 **Reporting Issues**
When reporting issues, please include:
- **Version used**: Which .py file
- **System**: macOS/Windows/Linux + Python version
- **Error message**: Full error output
- **Steps to reproduce**: What you were doing

---

## 📚 Additional Resources

### 🔗 **Related Technologies**
- [MediaPipe](https://mediapipe.dev/) - Google's ML framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [NumPy](https://numpy.org/) - Numerical computing


### 🎓 **Learning Resources**
- [MediaPipe Pose Guide](https://google.github.io/mediapipe/solutions/pose.html)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Computer Vision Basics](https://www.pyimagesearch.com/)

---

## 🙏 Acknowledgments

- **Google MediaPipe Team** - For excellent pose detection models
- **OpenCV Contributors** - For comprehensive computer vision tools
- **Research Community** - For virtual try-on algorithms and techniques

---

<div align="center">

### 🎉 **Ready to Try Virtual Fashion?**

**Choose your version and start the virtual try-on experience!**

[🚀 Quick Start](#-quick-start) • [💬 Report Issues](#-reporting-issues) • [📚 Learn More](#-additional-resources)

</div>
