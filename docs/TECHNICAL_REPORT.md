# Technical Report: Unified XAI Platform

**Project**: Multi-Modal Explainable AI Platform  
**Date**: January 2026  
**Team**: DIA5

---

## Executive Summary

This technical report documents the design, implementation, and integration of a unified Explainable AI platform that combines audio deepfake detection and medical image classification capabilities. The platform provides a seamless interface for multi-modal classification with integrated explainability methods (LIME, Grad-CAM, SHAP, Saliency Maps).

---

## 1. Project Objectives

### Primary Goals
1. Integrate two independent XAI repositories into a unified platform
2. Support multi-modal inputs (audio and images)
3. Implement multiple XAI methods with automatic compatibility filtering
4. Provide intuitive visualization and comparison capabilities

### Success Criteria
- ✅ Single unified interface for both modalities
- ✅ Automatic modality detection and XAI compatibility checking
- ✅ Side-by-side XAI comparison functionality
- ✅ Modular, extensible architecture
- ✅ Comprehensive documentation

---

## 2. Background & Context

### Original Repositories

#### Repository 1: Deepfake Audio Detection with XAI
- **Authors**: Aamir Hullur, Atharva Gurav, Aditi Govindu, Parth Godse
- **Purpose**: Detect synthetic/deepfake audio using CNNs
- **Dataset**: Fake-or-Real (FoR) dataset from York University
- **Models**: VGG16, MobileNet, ResNet, Custom CNN
- **Best Performance**: MobileNet (91.5% accuracy, 0.507 precision)
- **XAI Methods**: LIME, SHAP, Grad-CAM on mel-spectrograms
- **Implementation**: Jupyter notebooks + Streamlit app

#### Repository 2: Lung Cancer Detection
- **Author**: schaudhuri16
- **Purpose**: Detect malignant tumors in chest X-rays
- **Dataset**: CheXpert chest radiograph dataset
- **Models**: AlexNet, DenseNet (transfer learning)
- **Focus**: Binary classification (Benign vs Malignant)
- **XAI Method**: Grad-CAM for localization
- **Implementation**: Jupyter notebooks

### Integration Challenges
1. **Different frameworks**: Jupyter notebooks vs production code
2. **Different data types**: Audio signals vs medical images
3. **Different preprocessing**: Spectrogram generation vs image normalization
4. **Inconsistent interfaces**: Separate XAI implementations
5. **No unified user experience**: Two completely separate workflows

---

## 3. System Architecture

### 3.1 Overall Design Philosophy

The platform follows a **modular, layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│         User Interface (Streamlit)       │
│  - File Upload  - Model Selection        │
│  - Classification  - XAI Visualization   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│      Application Logic Layer             │
│  - Modality Detection                    │
│  - Workflow Orchestration                │
│  - Session Management                    │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
┌────────▼─────────┐  ┌──────▼──────────┐
│  Preprocessing   │  │  XAI Methods    │
│  - Audio         │  │  - LIME         │
│  - Image         │  │  - Grad-CAM     │
└────────┬─────────┘  │  - SHAP         │
         │            │  - Saliency     │
         │            └──────┬──────────┘
         │                   │
┌────────▼───────────────────▼──────────┐
│        Model Management Layer          │
│  - Model Registry                      │
│  - Model Loading                       │
│  - Prediction                          │
└────────────────────────────────────────┘
```

### 3.2 Module Breakdown

#### A. Utilities Module (`utils/`)

**1. preprocessing.py**
- `AudioPreprocessor`: Handles audio to spectrogram conversion
  - Loads audio files using librosa
  - Generates mel-spectrograms
  - Resizes to model input dimensions (224x224)
  - Normalizes pixel values
  
- `ImagePreprocessor`: Handles medical image preprocessing
  - Loads and resizes images
  - Applies model-specific normalization
  - Supports multiple preprocessing schemes (VGG16, ImageNet, etc.)

**2. model_loader.py**
- `ModelLoader`: Centralized model management
  - Model registry with metadata
  - Lazy loading and caching
  - Unified prediction interface
  - Support for dummy models (testing)

**3. compatibility.py**
- `XAICompatibilityChecker`: Ensures valid XAI method selection
  - Compatibility matrix for modality-XAI pairs
  - Dynamic filtering of incompatible methods
  - Method metadata and descriptions

#### B. XAI Methods Module (`xai_methods/`)

**1. lime_explainer.py**
- `LIMEExplainer`: LIME implementation
  - Superpixel-based perturbations
  - Local model approximation
  - Visualization with highlighted regions

**2. gradcam_explainer.py**
- `GradCAMExplainer`: Grad-CAM implementation
  - Automatic detection of last convolutional layer
  - Gradient computation using TensorFlow tape
  - Heatmap generation and overlay
  - Support for custom layer selection

**3. shap_explainer.py**
- `SHAPExplainer`: SHAP implementation
  - DeepExplainer for neural networks
  - Background dataset initialization
  - Pixel-level attribution
  - Multiple visualization modes

**4. saliency_explainer.py**
- `SaliencyExplainer`: Saliency Maps implementation
  - Vanilla gradient computation
  - Pixel-level sensitivity visualization
  - Support for binary and multi-class models
  - Fast and model-agnostic

#### C. User Interface (`app.py`)

**Pages**:
1. **Home**: Overview and quick start guide
2. **Classification & XAI**: Main workflow page
3. **XAI Comparison**: Side-by-side visualization
4. **About**: Team info, AI usage statement, references

**Key Features**:
- Session state management for workflow continuity
- Automatic modality detection from file extensions
- Dynamic UI updates based on input type
- Real-time feedback and error handling

---

## 4. Implementation Details

### 4.1 Audio Processing Pipeline

```python
Audio File (.wav)
    ↓
Load with librosa (sr=22050)
    ↓
Generate Mel-Spectrogram
    ↓
Convert to Image (224x224x3)
    ↓
Normalize [0, 1]
    ↓
Model Prediction
    ↓
XAI Explanation
```

**Technical Decisions**:
- **Sample Rate**: 22050 Hz (standard for speech/audio)
- **Spectrogram Type**: Mel-spectrogram (better for audio perception)
- **Image Size**: 224x224 (standard for CNN inputs)
- **Normalization**: [0, 1] range for consistency

### 4.2 Image Processing Pipeline

```python
Image File (.png/.jpg)
    ↓
Load with PIL
    ↓
Resize to 224x224
    ↓
Model-specific preprocessing
    ↓
Model Prediction
    ↓
XAI Explanation
```

**Technical Decisions**:
- **Color Space**: RGB (3 channels)
- **Normalization**: Model-dependent (ImageNet stats for transfer learning)
- **Augmentation**: Not applied during inference

### 4.3 XAI Compatibility System

The platform implements an intelligent compatibility matrix:

| XAI Method | Audio (Spectrogram) | Image (X-ray) | Requirements |
|------------|---------------------|---------------|--------------|
| LIME       | ✅                  | ✅            | None (model-agnostic) |
| Grad-CAM   | ✅                  | ✅            | Convolutional layers |
| SHAP       | ✅                  | ✅            | None (model-agnostic) || Saliency Maps | ✅               | ✅            | Differentiable model |
**Design Rationale**:
- All methods support both modalities since spectrograms are images
- Grad-CAM requires CNN architecture (verified at runtime)
- LIME and SHAP are model-agnostic
- Future methods can be added to the matrix easily

---

## 5. Model Selection & Performance

### 5.1 Audio Deepfake Detection Models

| Model | Accuracy | Precision | Recall | Parameters |
|-------|----------|-----------|--------|------------|
| **MobileNet** | **91.5%** | **0.507** | - | Lightweight, efficient |
| VGG16 | ~88% | - | - | Deep, high capacity |
| ResNet | ~89% | - | - | Residual connections |
| Custom CNN | ~87% | - | - | Tailored architecture |

**Selection Criteria**:
- MobileNet: Best accuracy and efficiency balance
- VGG16: Strong baseline, well-understood
- ResNet: Good for deep architectures
- Custom CNN: Flexibility for experimentation

### 5.2 Lung Cancer Detection Models

| Model | Architecture | Transfer Learning | Use Case |
|-------|--------------|-------------------|----------|
| **AlexNet** | 8 layers | ImageNet pre-trained | Fast inference |
| **DenseNet** | Dense connections | ImageNet pre-trained | High accuracy |
| VGG16 | 16 layers | ImageNet pre-trained | Robust baseline |

**Selection Criteria**:
- AlexNet: Efficient for real-time applications
- DenseNet: State-of-the-art performance with fewer parameters
- VGG16: Consistent, reliable performance

---

## 6. XAI Methods: Comparison & Analysis

### 6.1 LIME (Local Interpretable Model-agnostic Explanations)

**Mechanism**:
1. Segment image into superpixels
2. Generate perturbed samples by hiding regions
3. Train linear model on perturbations
4. Extract feature importance

**Strengths**:
- Model-agnostic (works with any classifier)
- Intuitive region-based explanations
- Local fidelity (accurate for specific instance)

**Weaknesses**:
- Computationally expensive (1000+ samples)
- Stochastic (different results on same input)
- Superpixel quality affects explanation

**Best For**: Understanding which regions matter for a specific prediction

### 6.2 Grad-CAM (Gradient-weighted Class Activation Mapping)

**Mechanism**:
1. Get predictions and feature maps from last conv layer
2. Compute gradients of target class w.r.t. feature maps
3. Weight feature maps by gradients
4. Generate heatmap showing important regions

**Strengths**:
- Fast (single forward + backward pass)
- Deterministic (same input → same output)
- Smooth, continuous heatmaps
- Class-discriminative

**Weaknesses**:
- Requires convolutional architecture
- Only uses last conv layer (may miss earlier features)
- Can highlight irrelevant regions if model is poorly trained

**Best For**: Localization - showing where the model "looks"

### 6.3 SHAP (SHapley Additive exPlanations)

**Mechanism**:
1. Based on game theory (Shapley values)
2. Compute contribution of each pixel/feature
3. Ensure fair attribution across all features
4. Generate pixel-level importance map

**Strengths**:
- Theoretically grounded (game theory)
- Model-agnostic (with appropriate explainer)
- Additive (contributions sum to prediction)
- Consistent and fair attribution

**Weaknesses**:
- Computationally intensive
- Requires background dataset
- Complex to interpret for high-dimensional inputs

**Best For**: Comprehensive, theoretically justified explanations

### 6.4 Saliency Maps (Vanilla Gradients)

**Mechanism**:
1. Compute gradients of model output w.r.t. input
2. Calculate gradient magnitude for each pixel
3. Visualize as heatmap showing sensitivity
4. Overlay on original input

**Strengths**:
- Extremely fast (single backward pass)
- Simple and interpretable
- Works with any differentiable model
- Pixel-level granularity
- No hyperparameters or background data needed

**Weaknesses**:
- Can be noisy without smoothing
- May highlight edges rather than objects
- Less class-discriminative than Grad-CAM
- Raw gradients can be difficult to interpret

**Best For**: Quick, pixel-level sensitivity analysis and feature importance

### 6.5 Comparison Table

| Aspect | LIME | Grad-CAM | SHAP | Saliency Maps |
|--------|------|----------|------|---------------|
| **Speed** | Slow | Fast | Medium | Very Fast |
| **Model-Agnostic** | ✅ | ❌ (needs CNN) | ✅ | ✅ (needs gradients) |
| **Deterministic** | ❌ | ✅ | ✅ | ✅ |
| **Granularity** | Superpixels | Spatial regions | Pixels | Pixels |
| **Theoretical Foundation** | Linear approximation | Gradient-based | Game theory | Gradient-based |
| **Interpretability** | High | Very High | Medium | Medium-High |

---

## 7. Key Improvements Over Original Repositories

### 7.1 Architecture & Code Quality

**Before** (Original Repos):
- Jupyter notebooks with mixed code and experiments
- Duplicated preprocessing logic
- No modularity or reusability
- Separate XAI implementations

**After** (Unified Platform):
- Production-ready Python modules
- Shared preprocessing utilities
- Reusable XAI method classes
- Clear separation of concerns

### 7.2 User Experience

**Before**:
- Need to run separate notebooks
- Manual file path management
- No visual comparison of XAI methods
- Technical knowledge required

**After**:
- Single web interface
- Drag-and-drop file upload
- Side-by-side XAI comparison
- User-friendly with guided workflow

### 7.3 Extensibility

**Before**:
- Adding new model = rewriting notebook
- Adding XAI method = separate implementation
- Hard to maintain consistency

**After**:
- New model = add to registry + file
- New XAI = implement interface + add to checker
- Automatic integration with UI

### 7.4 Functionality

**New Features**:
1. Automatic modality detection
2. Dynamic XAI compatibility filtering
3. Multi-XAI comparison view
4. Session state management with smart reset detection
5. Comprehensive error handling
6. Dummy models for testing
7. Saliency Maps for fast pixel-level attribution
8. Automatic Keras version detection (tf_keras vs keras 3.x)
9. Nested model support for transfer learning architectures
10. Persistent classification results during XAI generation

---

## 8. Technical Challenges & Solutions

### Challenge 1: Different Input Modalities
**Problem**: Audio and images require completely different preprocessing

**Solution**:
- Created separate preprocessor classes with common interface
- Spectrograms treated as images after generation
- Unified input shape (224x224x3) for all models

### Challenge 2: XAI Method Compatibility
**Problem**: Not all XAI methods work with all models/modalities

**Solution**:
- Implemented compatibility checker with metadata
- Dynamic UI filtering based on input type
- Clear error messages for unsupported combinations

### Challenge 3: Model Loading & Management
**Problem**: Different model formats and architectures from two repos

**Solution**:
- Created model registry with metadata
- Abstracted loading logic in ModelLoader class
- Support for both real and dummy models

### Challenge 4: Maintaining Separation Between Modalities
**Problem**: Keeping audio and image pipelines separate but using shared code

**Solution**:
- Modality-specific preprocessing classes
- Shared XAI implementations (work on all images)
- Configuration-based model selection

### Challenge 5: Streamlit State Management
**Problem**: Streamlit reruns entire script on interaction

**Solution**:
- Used `st.session_state` for persistence
- Cached expensive operations
- Structured workflow to minimize recomputation
- Implemented file/model change detection to reset predictions
- Persistent display of classification results

### Challenge 6: Keras Version Compatibility
**Problem**: Models saved with different Keras versions (tf_keras 2.x vs keras 3.x) causing GradCAM failures

**Solution**:
- Implemented automatic Keras version detection by inspecting model module
- Dynamic import of correct Keras module for gradient computation
- Nested model detection for transfer learning architectures
- Dictionary input format fallback for models with named inputs

### Challenge 7: Binary vs Multi-class Output Handling
**Problem**: Some models use single sigmoid output (binary), others use softmax (multi-class)

**Solution**:
- Output shape detection in XAI methods
- Conditional logic for binary (shape: `(None, 1)`) vs multi-class
- Proper gradient computation for both output types
- Consistent class index handling across methods

---

## 9. Testing & Validation

### 9.1 Unit Testing Approach

**Components Tested**:
- Preprocessing: Verified output shapes and normalization
- Model Loading: Tested registry lookups and loading
- Compatibility: Validated filtering logic
- XAI Methods: Checked output formats and visualizations

### 9.2 Integration Testing

**Workflows Tested**:
1. Audio upload → Classification → LIME
2. Image upload → Classification → Grad-CAM
3. Audio upload → Multiple XAI → Comparison
4. Model switching with same input
5. Error handling for invalid files

### 9.3 User Acceptance Testing

**Criteria**:
- ✅ Intuitive navigation
- ✅ Clear instructions and feedback
- ✅ Appropriate error messages
- ✅ Responsive performance
- ✅ Accurate explanations

---

## 10. Future Enhancements

### Short-term Improvements
1. **Additional Models**: Add more architectures (EfficientNet, Vision Transformer)
2. **Batch Processing**: Support multiple files at once
3. **Export Functionality**: Save explanations as reports (PDF, HTML)
4. **Performance Metrics**: Show model accuracy, F1 score, confusion matrix
5. **Smooth/Integrated Gradients**: Enhance saliency with smoothing techniques

### Medium-term Enhancements
1. **Additional Modalities**: Add video, text, or tabular data support
2. **More XAI Methods**: Integrated Gradients with smoothing, Attention visualization
3. **Custom Model Upload**: Allow users to upload their own models
4. **Dataset Management**: Built-in dataset loading and splitting
5. **Interactive Zoom**: Add Plotly or similar for interactive visualizations
6. **Model Comparison**: Side-by-side comparison of different models

### Long-term Vision
1. **Cloud Deployment**: Host as web service
2. **API Development**: RESTful API for programmatic access
3. **Collaborative Features**: Multi-user support, shared explanations
4. **AutoML Integration**: Automatic model selection and hyperparameter tuning

---

## 11. Lessons Learned

### Technical Insights
1. **Modularity is key**: Separating concerns made integration much easier
2. **Interface design matters**: Common interfaces allow swapping implementations
3. **Error handling is crucial**: Users will try unexpected inputs
4. **Documentation saves time**: Clear docstrings prevented confusion

### Project Management
1. **Start with architecture**: Planning before coding prevents rework
2. **Test incrementally**: Catching bugs early is much easier
3. **AI as assistant, not author**: Human oversight ensures quality
4. **User feedback is valuable**: Testing with real users reveals issues

---

## 12. Conclusion

This project successfully unified two independent XAI repositories into a cohesive, production-ready platform. The key achievements include:

1. **Seamless Integration**: Combined audio and image modalities in one interface
2. **Enhanced Usability**: Intuitive UI with automatic compatibility checking
3. **Comprehensive XAI**: Support for multiple explanation methods with comparison
4. **Extensible Design**: Easy to add new models, XAI methods, or modalities
5. **Production Quality**: Modular code, error handling, and documentation

The unified platform demonstrates how thoughtful architecture and careful integration can create a system that is greater than the sum of its parts. The use of generative AI accelerated development while maintaining code quality through human oversight and validation.

---

## References

1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should i trust you?" Explaining the predictions of any classifier. *KDD*.

2. Selvaraju, R. R., et al. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. *ICCV*.

3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.

4. Original Deepfake Audio Detection repository by Aamir Hullur et al.

5. Original Lung Cancer Detection repository by schaudhuri16

6. Streamlit Documentation: https://docs.streamlit.io/

7. TensorFlow/Keras Documentation: https://www.tensorflow.org/

---

**Document Version**: 1.1  
**Last Updated**: January 2026  
**Authors**: DIA5 Team
