# Quick Start Guide - Unified XAI Platform

Get up and running in 5 minutes! 🚀

## Prerequisites

- Python 3.8+ installed
- pip package manager
- ~2GB free disk space

## Installation Steps

### 1. Navigate to Project Directory

```bash
cd "C:\Users\renoi\OneDrive - DVHE\A5\Explainability AI\Project\unified-xai-platform"
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes depending on your internet connection.

### 4. Verify Installation

```bash
python -c "import streamlit; import tensorflow; import lime; print('All dependencies installed successfully!')"
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Quick Test

### Test with Audio (if you have .wav files):

1. Go to "Classification & XAI" page
2. Upload a .wav file
3. Select "MobileNet" model
4. Click "Run Classification"
5. Choose "LIME" from XAI methods
6. Click "Generate Explanation"

### Test with Images (if you have X-ray images):

1. Go to "Classification & XAI" page
2. Upload a chest X-ray image (.png or .jpg)
3. Select "DenseNet" model
4. Click "Run Classification"
5. Choose "Grad-CAM" from XAI methods
6. Click "Generate Explanation"

## Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution**: Make sure you activated the virtual environment and installed requirements.txt

### Problem: "Port 8501 already in use"
**Solution**: 
```bash
streamlit run app.py --server.port 8502
```

### Problem: "TensorFlow not found"
**Solution**: 
```bash
pip install tensorflow==2.12.0
```

### Problem: Models not loading
**Solution**: The platform uses dummy models for testing. For full functionality with trained models:
1. Place trained models in `models/audio/` or `models/image/`
2. Update model paths in `utils/model_loader.py` if needed

## Next Steps

1. Read the [README.md](../README.md) for comprehensive documentation
2. Check [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for implementation details
3. Review [AI_USAGE_STATEMENT.md](AI_USAGE_STATEMENT.md) for AI transparency

## Getting Sample Data

### Audio Samples:
- Use the original `Audio_fake/` and `Audio_real/` folders in the project
- Or record your own .wav files

### Image Samples:
- Use the original `LungCancerDetection/images/` folder
- Or download sample chest X-rays from public datasets

## Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Contact team members for assistance

---

**Enjoy exploring Explainable AI! 🔬🧠**
