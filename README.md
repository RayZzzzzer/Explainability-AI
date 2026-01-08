# Unified XAI Platform

A comprehensive multi-modal Explainable AI platform for audio deepfake detection and medical image classification with integrated explainability methods (LIME, Grad-CAM, SHAP).

## 👥 Team Information

**TD Group**: DIA5

**Team Members**:
- REDON Guillaume
- RENOIR Théo

## 📋 Project Overview

This project integrates two independent repositories into a unified platform:

1. **Deepfake Audio Detection with XAI** - Audio classification (Real vs Fake)
2. **Lung Cancer Detection** - Medical image classification (Benign vs Malignant)

### Key Features

- 🎯 **Multi-Modal Support**: Handles both audio (.wav) and image (chest X-ray) inputs
- 🧠 **Dynamic Model Loading**: Auto-discovers models from filesystem (SavedModel, .h5)
- 🔬 **Integrated XAI**: LIME, Grad-CAM, SHAP/Integrated Gradients
- ⚡ **Smart Filtering**: Automatic compatibility checking for XAI methods
- 📊 **Comparison View**: Side-by-side visualization of multiple XAI explanations
- 🎨 **Intuitive UI**: Clean Streamlit-based interface with drag-and-drop support

## 🏗️ Project Structure

```
Explainability-AI/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── preprocessing.py        # Audio/image preprocessing
│   ├── model_loader.py         # Model loading and management
│   └── compatibility.py        # XAI compatibility checker
│
├── xai_methods/                # XAI implementations
│   ├── __init__.py
│   ├── lime_explainer.py       # LIME implementation
│   ├── gradcam_explainer.py    # Grad-CAM implementation
│   └── shap_explainer.py       # SHAP implementation
│
├── models/                     # Trained models
│   ├── README.md               # Model documentation
│   ├── audio/                  # Audio models (auto-discovered)
│   │   ├── my_vgg16.h5         # VGG16 model (.h5 format)
│   │   └── mobilenet/          # MobileNet SavedModel (tf_keras)
│   └── image/                  # Image models (auto-discovered)
│       ├── custom_cnn_best.h5
│       └── transfer_learning_best.h5
│
├── data/                       # Data directory
│   ├── README.md
│   ├── audio_uploads/          # Uploaded audio files
│   └── image_uploads/          # Uploaded images
│
├── utils notebooks/            # Development notebooks
│   ├── convert_h5.ipynb        # Model conversion utilities
│   ├── convert_to_spectro.py   # Audio to spectrogram conversion
│   └── TrainModels.ipynb       # Model training notebooks
│
└── docs/                       # Documentation
    ├── TECHNICAL_REPORT.md     # Technical report
    ├── AI_USAGE_STATEMENT.md   # Generative AI usage declaration
    └── QUICK_START.md          # Quick start guide
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster inference

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/RayZzzzzer/Explainability-AI.git
cd Explainability-AI
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Pre-trained Models

Place your trained model files in the appropriate directories:

- **Audio models**: `models/audio/`
- **Image models**: `models/image/`

**Note**: The platform includes dummy models for testing. For full functionality, add your trained models.

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 How to Use

### Basic Workflow

1. **Navigate to Classification & XAI Page**
   - Upload an audio file (.wav) or chest X-ray image
   - The system automatically detects the input modality

2. **Select Model**
   - Choose from available models compatible with your input type
   - View model information and specifications

3. **Run Classification**
   - Click "Run Classification" to get predictions
   - View predicted class and confidence scores

4. **Generate XAI Explanations**
   - Select an XAI method (automatically filtered by compatibility)
   - Click "Generate Explanation" to visualize model decisions
   - Supported methods: LIME, Grad-CAM, SHAP, Saliency Maps

5. **Compare Explanations**
   - Navigate to "XAI Comparison" tab
   - View multiple XAI explanations side-by-side
   - Understand different perspectives on model decisions

### Supported Input Types

#### Audio Files
- **Format**: .wav
- **Use Case**: Deepfake audio detection
- **Classification**: Real vs Fake
- **Models**: MobileNet, VGG16, ResNet, Custom CNN
- **Preprocessing**: Automatic conversion to mel-spectrograms

#### Image Files
- **Format**: .png, .jpg, .jpeg, .bmp
- **Use Case**: Lung cancer detection in chest X-rays
- **Classification**: Benign vs Malignant
- **Models**: AlexNet, DenseNet, VGG16

### XAI Methods

| Method | Audio | Image | Description |
|--------|-------|-------|-------------|
| **LIME** | ✅ | ✅ | Local Interpretable Model-agnostic Explanations - highlights influential regions |
| **Grad-CAM** | ✅ | ✅ | Gradient-weighted Class Activation Mapping - visualizes model attention |
| **SHAP** | ✅ | ✅ | SHapley Additive exPlanations - game-theory based attribution |
| **Saliency Maps** | ✅ | ✅ | Vanilla Gradient Visualization - pixel-level sensitivity analysis |

## 🧪 Demo Instructions

### Audio Deepfake Detection Demo

1. Prepare sample audio files (real and fake)
2. Upload a .wav file through the interface
3. Select "MobileNet" model (best performance: 91.5% accuracy)
4. Run classification
5. Apply LIME or Grad-CAM to see which frequency regions influenced the decision
6. Compare multiple XAI methods in the comparison tab

### Lung Cancer Detection Demo

1. Prepare chest X-ray images
2. Upload an image file
3. Select "DenseNet" or "AlexNet" model
4. Run classification
5. Use Grad-CAM to visualize which anatomical regions the model focused on
6. Compare LIME and Grad-CAM to understand different explanation perspectives

## 🎯 Improvements Over Original Repositories

### Integration & Architecture
- **Unified codebase** replacing two separate projects
- **Modular design** with clear separation of concerns
- **Reusable components** for preprocessing, model loading, and XAI
- **Extensible framework** for adding new models or XAI methods

### User Experience
- **Single intuitive interface** for both modalities
- **Automatic compatibility filtering** prevents invalid XAI selections
- **Side-by-side comparison** of multiple explanations
- **Drag-and-drop file upload** with automatic modality detection

### Code Quality
- **Comprehensive documentation** with docstrings
- **Type hints** for better code clarity
- **Error handling** with user-friendly messages
- **Session state management** for smooth user flow

### Functionality
- **Multi-XAI support** in one platform
- **Real-time visualization** of explanations
- **Model metadata system** for easy extension
- **Dummy models** for testing without trained weights

## 🤖 Generative AI Usage Statement

### Declaration of AI Usage

This project was developed with assistance from generative AI tools in accordance with academic integrity guidelines.

### Tools Used

- **GitHub Copilot (Claude Sonnet 4.5)**
  - Primary AI assistant for code generation and refactoring

### How AI Was Used

1. **Code Refactoring**
   - Restructuring original repositories into modular architecture
   - Converting Jupyter notebooks to production-ready Python modules
   - Implementing design patterns and best practices

2. **Architecture Design**
   - Planning the unified platform structure
   - Designing the compatibility checking system
   - Creating modular component interfaces

3. **Documentation**
   - Writing comprehensive README documentation
   - Creating technical report
   - Generating code comments and docstrings

4. **Implementation**
   - XAI method wrapper implementations
   - Streamlit UI components
   - Error handling and edge cases

### Human Contributions

- **Requirements Analysis**: Defining project scope and functional requirements
- **Design Decisions**: Choosing architecture patterns and technology stack
- **Integration Strategy**: Determining how to merge two different repositories
- **Testing & Validation**: Verifying functionality and user experience
- **Critical Review**: Ensuring code quality, correctness, and academic standards
- **Customization**: Adapting AI-generated code to specific project needs

### Ethical Considerations

All AI-generated code was:
- Reviewed and understood by team members
- Modified to fit project requirements
- Tested for correctness and functionality
- Attributed appropriately in this document

## 📚 Technical Details

### Model Architectures

#### Audio Classification
- Input: Mel-spectrograms (224x224x3)
- Architecture: CNNs (VGG16, MobileNet, ResNet, Custom)
- Output: Binary classification (Real/Fake)

#### Image Classification
- Input: Chest X-ray images (224x224x3)
- Architecture: CNNs (AlexNet, DenseNet, VGG16)
- Output: Binary classification (Benign/Malignant)

### XAI Implementation Details

**LIME (Local Interpretable Model-agnostic Explanations)**:
- Uses superpixel segmentation
- Perturbs input regions to measure impact
- Provides local explanations

**Grad-CAM (Gradient-weighted Class Activation Mapping)**:
- Computes gradients of target class w.r.t. feature maps
- Generates heatmap highlighting important regions
- Requires convolutional layers
- Uses tf_keras for compatibility with SavedModel format
- **Automatic keras version detection** - adapts to model's keras version
- **Nested model support** - handles transfer learning models (MobileNet, VGG16, etc.)

**SHAP (SHapley Additive exPlanations)**:
- Tries GradientExplainer and DeepExplainer first
- Falls back to **Integrated Gradients** for tf_keras models
- Provides pixel-level attributions
- Fast and reliable gradient-based explanations

**Saliency Maps (Vanilla Gradients)**:
- Computes gradients of output w.r.t. input
- Shows pixel-level sensitivity/importance
- Simple and fast computation
- Works with any differentiable model
- Provides fine-grained feature attribution

### Key Technical Components

**tf_keras Compatibility**:
- Uses `tensorflow.keras` (tf_keras 2.20.1) for legacy SavedModel support
- Keras 3.13.0 for new model development
- Dual compatibility layer for maximum model support

**Dynamic Model Discovery**:
- Automatically scans `models/{modality}/` directory
- Supports both SavedModel format and .h5 files
- No hardcoded model registry required

**Clean Spectrogram Generation**:
- Removes matplotlib axes, labels, and decorations
- Generates pure spectrogram images for better XAI visualization

## 🐛 Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError" when running the app
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Models not loading
- **Solution**: The platform uses dummy models by default. Add trained models to `models/` directory for full functionality

**Issue**: SHAP installation fails
- **Solution**: Try `pip install shap --no-cache-dir` or use a different Python version (3.8-3.10 recommended)

**Issue**: GradCAM fails with "Cannot create gradient model" error
- **Solution**: This can happen with certain model architectures. The system will automatically try alternative approaches. If it persists, use LIME or SHAP instead

**Issue**: Streamlit warnings about `use_container_width`
- **Solution**: These are deprecation warnings and don't affect functionality. Update to `width='stretch'` if desired

## 📄 License

This project integrates work from:
- Deepfake Audio Detection with XAI (original authors: Aamir Hullur, et al.)
- Lung Cancer Detection (original author: schaudhuri16)

Please respect the licenses of the original repositories.

**Last Updated**: January 2025  
**Version**: 1.0.0
