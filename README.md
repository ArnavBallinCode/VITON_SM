# 🎽 Virtual Shirt Try-On System

<div align="center">

![Virtual Try-On Demo](https://img.shields.io/badge/Status-Production_Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-yellow)

**Professional AI-powered virtual try-on system using advanced computer vision and pose estimation**

[🚀 Quick Start](#-quick-start) • [📋 Features](#-features) • [🛠️ Installation](#️-installation) • [📱 Usage](#-usage) • [🔧 Troubleshooting](#-troubleshooting)

</div>

---

## 📋 Features

### 🎯 **Two Professional Implementations**

#### **🌟 Simple Perfect Try-On v1 (Advanced Background Removal)**
- **Advanced garment background removal** with multi-method detection
- **Professional alpha blending** with edge smoothing
- **Corner-based background sampling** for intelligent detection
- **Edge feathering** for natural appearance
- **High-quality LANCZOS4 interpolation**

#### **✨ Simple Perfect Try-On v2 (Real-time Segmentation)**
- **Live background removal** using MediaPipe Selfie Segmentation
- **Real-time person isolation** from background
- **Seamless garment overlay** on isolated subject
- **Optimized performance** for smooth real-time operation
- **Clean, professional output**

### 🌟 **Core Capabilities**
- **✅ Real-time pose detection** using MediaPipe Pose (33 landmarks)
- **🎭 Intelligent background removal** (dual approaches)
- **👕 Smart garment fitting** based on body measurements
- **🎨 Professional alpha compositing** for realistic overlay
- **📸 High-quality screenshot capture** with organized storage
- **🔄 Live garment switching** during runtime
- **🪞 Mirror mode** for natural interaction experience

### 🔧 **Technical Excellence**
- **Multi-threaded processing** for optimal performance
- **Adaptive sizing algorithms** based on shoulder width and torso height
- **Edge-preserving smoothing** for natural garment integration
- **Multi-format support** (PNG with transparency, JPG, JPEG)
- **Automatic garment detection** from organized folder structure
- **Professional error handling** with graceful degradation

---

## 🛠️ Installation

### 📋 Prerequisites
- **Python 3.8+** (Recommended: Python 3.9+)
- **Webcam** or USB camera device
- **Operating System**: macOS, Windows 10/11, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 500MB free space

### 🔧 Required Dependencies

**Option 1: Using requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option 2: Manual installation**
```bash
pip install opencv-python>=4.5.0
pip install mediapipe>=0.8.10
pip install numpy>=1.21.0
pip install scipy>=1.7.0
```

### ✅ Verify Installation
```bash
python -c "import cv2, mediapipe, numpy, scipy; print('✅ All dependencies installed successfully')"
```

### 📁 Project Structure
```
Virtual-Shirt-Try-On/
├── 📄 README.md                    # This documentation
├── 📄 requirements.txt             # Python dependencies
├── 🐍 simple_perfect_tryon_1.py    # Version 1: Advanced background removal
├── 🐍 simple_perfect_tryon_2.py    # Version 2: Real-time segmentation
├── 📁 Garments/
│   └── 📁 tops/                   # Shirt/top garments
│       ├── 🖼️ 1.png
│       ├── 🖼️ 3.png
│       └── 🖼️ 4.png
├── 📁 Shirts/                     # Additional garments folder
│   ├── 🖼️ 1.png
│   ├── 🖼️ 3.png
│   └── 🖼️ 4.png
├── 📁 Screenshots/                # Auto-generated screenshots
└── 📁 different test versions/    # Archive of development versions
```

---

## 🚀 Quick Start

### 1️⃣ **Choose Your Version**

#### 🌟 **Version 1: Advanced Background Removal**
```bash
python simple_perfect_tryon_1.py
```
- **Best for**: Professional results, high-quality output
- **Technology**: Advanced garment background removal algorithms
- **Features**: Multi-method detection, edge feathering, corner sampling
- **Output**: Studio-quality garment integration

#### ✨ **Version 2: Real-time Segmentation**
```bash
python simple_perfect_tryon_2.py
```
- **Best for**: Real-time performance, live demonstrations
- **Technology**: MediaPipe Selfie Segmentation
- **Features**: Live background removal, optimized processing
- **Output**: Clean, real-time virtual try-on experience

### 2️⃣ **First Run**
1. **Connect your camera** and ensure it's working
2. **Place garment images** in `Garments/tops/` or `Shirts/` folder
3. **Run your chosen version** using the commands above
4. **Stand 6-8 feet** from the camera for optimal detection
5. **Use keyboard controls** to interact with the system

---

## 📱 Usage

### 🎮 **Keyboard Controls**
| Key | Action | Description |
|-----|--------|-------------|
| **N** | Next Garment | Switch to next available garment |
| **P** | Previous Garment | Switch to previous garment |
| **S** | Save Screenshot | Capture current view to Screenshots/ folder |
| **ESC** | Exit | Close the application safely |

### 📸 **Screenshot Management**
- **Auto-organization**: Screenshots automatically saved to `Screenshots/` folder
- **Naming convention**: `perfect_tryon_{timestamp}.jpg`
- **High quality**: Full resolution captures with professional compression
- **Examples**:
  - `Screenshots/perfect_tryon_1749108693.jpg`
  - `Screenshots/perfect_tryon_1749108734.jpg`

### 👕 **Adding New Garments**

#### **Supported Formats**
- **PNG**: Recommended for garments with transparency
- **JPG/JPEG**: Standard images (transparency will be automatically detected)

#### **Optimal Garment Preparation**
- **Background**: White, light gray, or transparent backgrounds work best
- **Resolution**: Higher resolution images produce better results (minimum 500x500px)
- **Quality**: Clean, well-lit garment images without wrinkles
- **Orientation**: Garments should be front-facing and centered

#### **Adding Process**
1. **Prepare your garment images**:
   ```bash
   # Recommended specifications
   - Format: PNG with transparency or JPG with white background
   - Size: 1000x1000px or higher
   - Background: White/transparent for best results
   ```

2. **Add to designated folders**:
   ```bash
   # Copy to tops folder
   cp your_shirt.png Garments/tops/
   
   # Or to Shirts folder
   cp your_shirt.png Shirts/
   ```

3. **Restart the application** - new garments are automatically detected

### 🎯 **Optimal Usage Guidelines**

#### 📷 **Camera Setup**
- **Distance**: Position yourself 6-8 feet from the camera
- **Height**: Camera should be at chest/shoulder level
- **Angle**: Face the camera directly for best pose detection
- **Lighting**: Use even, natural lighting (avoid harsh shadows)
- **Background**: Plain, contrasting background recommended

#### 🧍 **Body Positioning**
- **Posture**: Stand upright with shoulders square to camera
- **Arms**: Keep arms slightly away from your body
- **Visibility**: Ensure full upper torso is visible in frame
- **Movement**: Minimize sudden movements for stable tracking
- **Clothing**: Wear fitted clothing for accurate body detection

#### 🎨 **Performance Optimization**
- **Close unnecessary applications** to free up system resources
- **Use good lighting** to improve pose detection accuracy
- **Ensure stable internet** (not required but helps with updates)
- **Keep garment folders organized** for easy navigation

---

## 🔧 Technical Architecture

### 🎯 **Version 1: Advanced Background Removal**

#### **Core Technologies**
- **MediaPipe Pose**: 33-landmark body detection with high precision
- **Advanced Background Removal**: Multi-method garment background detection
- **Professional Alpha Blending**: Edge-preserving composition techniques
- **High-Quality Interpolation**: LANCZOS4 for superior image quality

#### **Background Removal Pipeline**
```python
# Multi-method detection process
1. White/Light Background Detection → Threshold-based mask
2. Corner Color Sampling → Intelligent background color detection  
3. Edge Feathering → Smooth, natural transitions
4. Alpha Channel Enhancement → Professional transparency
5. Gaussian Smoothing → Anti-aliasing and edge refinement
```

#### **Advanced Features**
- **Corner-based sampling**: Analyzes image corners to detect background color
- **Multi-threshold detection**: Uses multiple threshold values for accuracy
- **Edge feathering**: Creates natural, soft edges around garments
- **Professional alpha blending**: Industry-standard composition techniques
- **Error handling**: Graceful degradation with invalid inputs

### ✨ **Version 2: Real-time Segmentation**

#### **Core Technologies**
- **MediaPipe Pose**: Real-time body landmark detection
- **MediaPipe Selfie Segmentation**: Advanced person isolation from background
- **Optimized Processing**: Streamlined pipeline for real-time performance
- **Live Composition**: Real-time background replacement and garment overlay

#### **Segmentation Pipeline**
```python
# Real-time processing workflow
1. Live Video Capture → High-resolution camera input
2. Person Segmentation → AI-powered background removal
3. Pose Detection → 33-landmark body tracking
4. Garment Positioning → Smart size and position calculation
5. Real-time Composition → Live overlay and blending
```

#### **Performance Features**
- **Optimized processing**: Streamlined for 30+ FPS performance
- **Memory efficient**: Minimal memory footprint for smooth operation
- **Live background removal**: Real-time person isolation
- **Adaptive quality**: Automatic quality adjustment based on system performance

### 🎨 **Shared Core Features**

#### **Pose Detection System**
- **33 Landmark Detection**: Full body pose analysis using MediaPipe
- **Key Points Used**:
  - Shoulders (left/right) for width calculation
  - Hips (left/right) for torso length measurement  
  - Center point calculation for accurate positioning
- **Adaptive Sizing**: Garment size scales with detected body proportions
- **Real-time Tracking**: Smooth, responsive pose following

#### **Smart Garment Fitting**
```python
# Intelligent sizing algorithm
shoulder_width = distance(left_shoulder, right_shoulder)
torso_height = distance(shoulder_center, hip_center)

garment_width = shoulder_width * 1.4   # 40% wider than shoulders
garment_height = torso_height * 1.2    # 20% longer than torso
```

#### **Professional Image Processing**
- **LANCZOS4 Interpolation**: Highest quality image resizing
- **Bounds Checking**: Prevents garments from extending outside frame
- **Alpha Channel Processing**: Professional transparency handling
- **Color Space Optimization**: Optimized BGR/RGB conversions

---

## 🔧 Advanced Configuration

### ⚙️ **Camera Settings**
```python
# Customizable in source code
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Default: 1280px
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Default: 720px
# Recommended: 1920x1080 for high-quality, 1280x720 for performance
```

### 🎯 **MediaPipe Configuration**
```python
# Pose detection settings (both versions)
model_complexity=1                    # 0=Light, 1=Full, 2=Heavy
min_detection_confidence=0.5          # Range: 0.0-1.0
min_tracking_confidence=0.5           # Range: 0.0-1.0
smooth_landmarks=True                 # Enable landmark smoothing
```

### 🎨 **Background Customization (Version 2)**
```python
# Custom background options
background = np.zeros(frame.shape, dtype=np.uint8)  # Black (default)
# Or load custom background:
# background = cv2.imread('custom_background.jpg')
# background = cv2.resize(background, (w, h))
```

### 📊 **Performance Tuning**
```python
# For slower systems
model_complexity=0                    # Lighter pose model
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# For high-end systems  
model_complexity=2                    # Heavy pose model for max accuracy
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 4K resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
```

---

## 🔧 Troubleshooting

### ❌ **Common Issues & Solutions**

#### 🚫 **"Cannot access camera"**
**Problem**: Camera permission denied or device not found
```bash
# Solution 1: Check camera permissions
# macOS: System Preferences > Security & Privacy > Camera
# Windows: Settings > Privacy > Camera
# Linux: Check camera device permissions

# Solution 2: Test camera independently
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera accessible:', cap.isOpened())"

# Solution 3: Try different camera indices
# Modify in source code: cv2.VideoCapture(1) or cv2.VideoCapture(2)
```

#### 🚫 **"No garments found"**
**Problem**: Garment folder empty or images not recognized
```bash
# Verify folder structure
ls -la Garments/tops/
ls -la Shirts/

# Check supported formats
# Supported: .png, .jpg, .jpeg (case insensitive)

# Add sample garments
cp sample_shirt.png Garments/tops/
```

#### 🚫 **"Module not found" errors**
**Problem**: Missing Python dependencies
```bash
# Solution 1: Install all requirements
pip install -r requirements.txt

# Solution 2: Install individually  
pip install opencv-python mediapipe numpy scipy

# Solution 3: Verify installation
python -c "import cv2, mediapipe, numpy, scipy; print('✅ All modules installed')"

# Solution 4: Use virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### 🚫 **Poor pose detection**
**Problem**: Body not detected or inaccurate tracking
```bash
# Solutions:
1. Improve lighting - use even, natural light
2. Use plain background - avoid busy patterns
3. Adjust distance - stand 6-8 feet from camera
4. Wear fitted clothing - loose clothing affects detection
5. Ensure full visibility - upper body fully in frame
6. Check camera angle - camera at chest level works best
```

#### 🚫 **Garment appears incorrectly**
**Problem**: Garment upside-down, too large/small, or poorly positioned
```bash
# Version 1 Solutions:
- Check garment image orientation (should be front-facing)
- Ensure white/transparent background in garment image
- Try different garment images
- Adjust body position relative to camera

# Version 2 Solutions:  
- Improve person segmentation by using better lighting
- Stand against contrasting background
- Ensure stable pose for consistent detection
```

#### 🚫 **Performance issues (slow/laggy)**
**Problem**: Low frame rate or system overload
```bash
# Solution 1: Reduce camera resolution (in source code)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Solution 2: Lower MediaPipe complexity
model_complexity=0  # Lightest model

# Solution 3: Close other applications
# Solution 4: Use Version 2 for better performance
python simple_perfect_tryon_2.py
```

### 🛠️ **Advanced Troubleshooting**

#### 🔍 **Enable Debug Mode**
Add this to the beginning of either script for detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
print("🔍 Debug mode enabled")
```

#### 📊 **System Requirements Check**
```bash
# Check Python version
python --version  # Should be 3.8+

# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')"

# Check camera devices
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(4)]"
```

#### 🔧 **Performance Monitoring**
```python
# Add to main loop for FPS monitoring
import time
frame_count = 0
start_time = time.time()

# In main loop:
frame_count += 1
if frame_count % 30 == 0:
    fps = frame_count / (time.time() - start_time)
    print(f"Current FPS: {fps:.1f}")
```

### 🚨 **Error Codes & Solutions**

| Error | Cause | Solution |
|-------|-------|----------|
| `(-215:Assertion failed)` | Invalid image dimensions | Check garment image format and size |
| `AttributeError: module 'cv2'` | OpenCV version mismatch | `pip install --upgrade opencv-python` |
| `ImportError: No module named 'mediapipe'` | Missing MediaPipe | `pip install mediapipe` |
| `Permission denied (camera)` | Camera access blocked | Grant camera permissions in system settings |
| `IndexError: list index out of range` | No garments found | Add images to Garments/tops/ or Shirts/ |
| `Memory allocation error` | Insufficient RAM | Close other applications, reduce resolution |

### 📞 **Getting Help**

#### **Before Reporting Issues**
1. ✅ **Test both versions** - try the other version if one fails
2. ✅ **Check system requirements** - ensure Python 3.8+ and sufficient RAM  
3. ✅ **Verify dependencies** - run installation verification command
4. ✅ **Test with sample images** - use provided garment samples
5. ✅ **Check camera independently** - test camera with other applications

#### **Reporting Issues**
When reporting problems, please include:
- **System**: macOS/Windows/Linux version
- **Python version**: `python --version`
- **Error message**: Full error output
- **Steps to reproduce**: What you were doing when error occurred
- **Version used**: simple_perfect_tryon_1.py or simple_perfect_tryon_2.py

---

## 📚 Technical Reference

### 🔗 **Core Technologies**
- [**MediaPipe**](https://mediapipe.dev/) - Google's ML framework for pose detection
- [**OpenCV**](https://opencv.org/) - Computer vision and image processing
- [**NumPy**](https://numpy.org/) - Numerical computing for Python
- [**SciPy**](https://scipy.org/) - Scientific computing (Version 1 only)

### 🎓 **Learning Resources**
- [MediaPipe Pose Documentation](https://google.github.io/mediapipe/solutions/pose.html)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Computer Vision Fundamentals](https://www.pyimagesearch.com/)
- [Alpha Blending Techniques](https://en.wikipedia.org/wiki/Alpha_compositing)

### 📊 **Performance Benchmarks**

#### **Version 1 (Advanced Background Removal)**
- **Processing**: ~15-25 FPS on modern hardware
- **Memory usage**: ~200-300 MB
- **CPU usage**: Medium (background processing intensive)
- **Best for**: High-quality output, professional results

#### **Version 2 (Real-time Segmentation)**  
- **Processing**: ~25-35 FPS on modern hardware
- **Memory usage**: ~150-250 MB
- **CPU usage**: Low-Medium (optimized pipeline)
- **Best for**: Real-time demos, smooth performance

#### **Recommended Hardware**
- **Minimum**: Intel i5 / AMD Ryzen 5, 4GB RAM, USB 2.0 camera
- **Recommended**: Intel i7 / AMD Ryzen 7, 8GB RAM, USB 3.0 1080p camera
- **Optimal**: Intel i9 / AMD Ryzen 9, 16GB RAM, High-quality webcam

---

## 🤝 Development & Customization

### 🔧 **Customization Options**

#### **Modify Garment Sizing**
```python
# In main() function, adjust sizing factors:
garment_width = max(int(shoulder_width * 1.4), 200)   # Change 1.4 to adjust width
garment_height = max(int(torso_height * 1.2), 250)    # Change 1.2 to adjust height
```

#### **Change Background Color (Version 2)**
```python
# Replace black background with custom color:
background = np.full(frame.shape, (15, 45, 85), dtype=np.uint8)  # Dark blue
# Or load image background:
# background = cv2.imread('background_image.jpg')
# background = cv2.resize(background, (w, h))
```

#### **Adjust Detection Sensitivity**
```python
# Make pose detection more/less sensitive:
min_detection_confidence=0.7  # Higher = more selective (0.5 default)
min_tracking_confidence=0.7   # Higher = more stable tracking (0.5 default)
```

### 🧪 **Development Setup**
```bash
# Clone/download project
cd Virtual-Shirt-Try-On

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # macOS/Linux
# dev_env\Scripts\activate   # Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Additional dev tools

# Run tests
python -c "import cv2, mediapipe; print('✅ Development environment ready')"
```

### 🔄 **Code Structure**

#### **Version 1 Architecture**
```python
# Key functions in simple_perfect_tryon_1.py:
remove_garment_background()      # Advanced background removal
enhance_garment_alpha_blending() # Professional alpha compositing
main()                          # Main application loop
```

#### **Version 2 Architecture**  
```python
# Key sections in simple_perfect_tryon_2.py:
mp_selfie_segmentation          # Real-time person segmentation
pose.process()                  # Pose detection pipeline
alpha blending section          # Real-time garment overlay
```

---

## 🙏 Acknowledgments

- **Google MediaPipe Team** - For exceptional pose detection and segmentation models
- **OpenCV Community** - For comprehensive computer vision tools and documentation
- **Python Community** - For NumPy, SciPy, and ecosystem support
- **Computer Vision Research Community** - For foundational algorithms in virtual try-on technology

---

<div align="center">

### 🎉 **Ready for Professional Virtual Try-On?**

**Choose your version and experience the future of virtual fashion!**

[🚀 Get Started](#-quick-start) • [💬 Report Issues](#-getting-help) • [📚 Learn More](#-learning-resources)

**Built with ❤️ using cutting-edge AI and computer vision**

</div>
