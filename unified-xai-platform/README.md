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
- 🧠 **Multiple Models**: VGG16, MobileNet, ResNet, Custom CNN (audio), AlexNet, DenseNet (images)
- 🔬 **Integrated XAI**: LIME, Grad-CAM, and SHAP explanations
- ⚡ **Smart Filtering**: Automatic compatibility checking for XAI methods
- 📊 **Comparison View**: Side-by-side visualization of multiple XAI explanations
- 🎨 **Intuitive UI**: Clean Streamlit-based interface with drag-and-drop support

## 🏗️ Project Structure

```
unified-xai-platform/
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
├── models/                     # Trained models (not included)
│   ├── audio/                  # Audio classification models
│   └── image/                  # Image classification models
│
├── data/                       # Data directory
│   ├── audio_uploads/          # Uploaded audio files
│   └── image_uploads/          # Uploaded images
│
└── docs/                       # Documentation
    ├── TECHNICAL_REPORT.md     # Technical report
    └── AI_USAGE_STATEMENT.md   # Generative AI usage declaration
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster inference

### Step 1: Clone or Download the Repository

```bash
cd unified-xai-platform
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
   - Supported methods: LIME, Grad-CAM, SHAP

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

**LIME**:
- Uses superpixel segmentation
- Perturbs input regions to measure impact
- Provides local explanations

**Grad-CAM**:
- Computes gradients of target class w.r.t. feature maps
- Generates heatmap highlighting important regions
- Requires convolutional layers

**SHAP**:
- Based on Shapley values from game theory
- Provides pixel-level attributions
- Model-agnostic approach

## 🐛 Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError" when running the app
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: "No module named 'utils'" error
- **Solution**: Make sure you're running the app from the `unified-xai-platform` directory

**Issue**: Models not loading
- **Solution**: The platform uses dummy models by default. Add trained models to `models/` directory for full functionality

**Issue**: SHAP installation fails
- **Solution**: Try `pip install shap --no-cache-dir` or use a different Python version (3.8-3.10 recommended)

## 📄 License

This project integrates work from:
- Deepfake Audio Detection with XAI (original authors: Aamir Hullur, et al.)
- Lung Cancer Detection (original author: schaudhuri16)

Please respect the licenses of the original repositories.

**Last Updated**: January 2025  
**Version**: 1.0.0
