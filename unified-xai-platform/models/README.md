# Model Configuration

This directory should contain trained models for the platform.

## Directory Structure

```
models/
├── audio/              # Audio classification models
│   ├── mobilenet.h5
│   ├── vgg16_audio.h5
│   ├── resnet_audio.h5
│   └── custom_cnn_audio.h5
│
└── image/              # Image classification models
    ├── alexnet.h5
    ├── densenet.h5
    └── vgg16_image.h5
```

## Where to Get Models

### Option 1: Use Original Repository Models

Copy trained models from the original repositories:

**Audio Models**:
- Source: `Deepfake-Audio-Detection-with-XAI/Streamlit/saved_model/`
- Copy to: `unified-xai-platform/models/audio/`

**Image Models**:
- Train using notebooks in `LungCancerDetection/`
- Save to: `unified-xai-platform/models/image/`

### Option 2: Train Your Own

Use the Jupyter notebooks in the original repositories to train models:

1. **Audio Models**: 
   - `Code/VGG16-Custom CNN-ResNet.ipynb`
   - `Code/InceptionV3-MobileNet.ipynb`

2. **Image Models**:
   - Use transfer learning with ImageNet pre-trained weights
   - Fine-tune on CheXpert dataset

### Option 3: Use Dummy Models (Testing Only)

The platform automatically creates dummy models if real models are not found. These are for testing the interface only and will not provide meaningful results.

## Model File Formats

Supported formats:
- `.h5` (Keras HDF5 format)
- SavedModel format (directory with `saved_model.pb`)

## Model Requirements

All models should:
- Accept input shape: (None, 224, 224, 3)
- Output shape: (None, 2) for binary classification
- Use softmax activation on final layer

## Copying Models

### From Original Audio Model

```bash
# Windows
xcopy "..\Deepfake-Audio-Detection-with-XAI\Streamlit\saved_model\model" "models\audio\mobilenet\" /E /I

# Linux/Mac
cp -r ../Deepfake-Audio-Detection-with-XAI/Streamlit/saved_model/model models/audio/mobilenet/
```

Then rename to match the expected format in `utils/model_loader.py`.

## Notes

- Models are not included in git (see `.gitignore`)
- Large model files should be stored separately
- Consider using Git LFS for version control of models
- The platform will work with dummy models for demonstration purposes
