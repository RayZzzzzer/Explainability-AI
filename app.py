"""
Unified XAI Platform - Main Streamlit Application

A multi-modal explainable AI platform for audio deepfake detection 
and medical image classification with integrated XAI methods.
"""

import streamlit as st
import numpy as np
import os
import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.preprocessing import AudioPreprocessor, ImagePreprocessor
from utils.model_loader import ModelLoader
from utils.compatibility import XAICompatibilityChecker
from xai_methods.lime_explainer import LIMEExplainer
from xai_methods.gradcam_explainer import GradCAMExplainer
from xai_methods.shap_explainer import SHAPExplainer

# Page configuration
st.set_page_config(
    page_title="Unified XAI Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
DEFAULT_SESSION_STATE = {
    'uploaded_file': None,
    'modality': None,
    'preprocessed_input': None,
    'predictions': None,
    'predictions_2class': None,
    'predicted_class': None,
    'class_names': None,
    'selected_model': None,
    'xai_results': {}
}

for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def detect_modality(file) -> str:
    """
    Detect input modality based on file extension.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        'audio', 'image', or None if unsupported
    """
    filename = file.name.lower()
    if filename.endswith('.wav'):
        return 'audio'
    elif filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        return 'image'
    return None


def main() -> None:
    """Main application entry point - handles page navigation and routing."""
    # Sidebar navigation
    st.sidebar.title("ESILV DIA - XAI PROJECT : Lung Cancer & Audio Deepfake Detection")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Classification & XAI", "XAI Comparison"]
    )
    
    if page == "Classification & XAI":
        show_classification_page()
    elif page == "XAI Comparison":
        show_comparison_page()



def show_classification_page():
    """Main classification and XAI page"""
    st.title("Classification & Explainability")
    
    # File upload section
    st.markdown("### 1 - Upload Input File")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload an audio file (.wav) or chest X-ray image",
            type=['wav', 'png', 'jpg', 'jpeg', 'bmp'],
            help="Drag and drop a file or click to browse"
        )
    
    if uploaded_file is not None:
        # Check if file has changed - reset predictions if so
        if st.session_state.uploaded_file is None or uploaded_file.name != st.session_state.uploaded_file.name:
            # New file uploaded - clear previous results
            st.session_state.predictions = None
            st.session_state.predictions_2class = None
            st.session_state.predicted_class = None
            st.session_state.class_names = None
            st.session_state.preprocessed_input = None
            st.session_state.xai_results = {}
        
        st.session_state.uploaded_file = uploaded_file
        st.session_state.modality = detect_modality(uploaded_file)
        
        with col2:
            if st.session_state.modality == 'audio':
                st.success("✅ Audio file detected")
                st.audio(uploaded_file)
            elif st.session_state.modality == 'image':
                st.success("✅ Image file detected")
                from PIL import Image
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_container_width=True)
            else:
                st.error("❌ Unsupported file type")
                return
        
        st.markdown("---")
        
        # Model selection
        st.markdown("### 2 - Select Model")
        
        model_loader = ModelLoader(models_dir='models')
        available_models = model_loader.get_available_models(st.session_state.modality)
        
        model_options = {
            info['name']: (key, info) 
            for key, info in available_models.items()
        }
        
        selected_model_name = st.selectbox(
            f"Choose a {st.session_state.modality} classification model:",
            options=list(model_options.keys()),
            help="Select the model architecture for classification"
        )
        
        selected_model_key, model_info = model_options[selected_model_name]
        
        # Check if model has changed - reset predictions if so
        if st.session_state.selected_model != selected_model_key:
            st.session_state.predictions = None
            st.session_state.predictions_2class = None
            st.session_state.predicted_class = None
            st.session_state.class_names = None
            st.session_state.preprocessed_input = None
            st.session_state.xai_results = {}
            st.session_state.selected_model = selected_model_key
        
        with st.expander("ℹ️ Model Information"):
            st.write(f"**Description**: {model_info['description']}")
            st.write(f"**Input Shape**: {model_info['input_shape']}")
            st.write(f"**Classes**: {', '.join(model_info['classes'])}")
        
        st.markdown("---")
        
        # Classification
        st.markdown("###  3 - Classify Input")
        
        if st.button("Run Classification", type="primary"):
            with st.spinner("Processing input and running classification..."):
                try:
                    # Preprocess input
                    if st.session_state.modality == 'audio':
                        preprocessor = AudioPreprocessor()
                        
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Create spectrogram
                        spec_image = preprocessor.create_spectrogram(tmp_path)
                        preprocessed = preprocessor.preprocess_for_model(spec_image)
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                        st.session_state.preprocessed_input = spec_image
                        
                    else:  # image
                        preprocessor = ImagePreprocessor()
                        from PIL import Image
                        img = Image.open(uploaded_file)
                        preprocessed = preprocessor.preprocess_for_model(img)
                        st.session_state.preprocessed_input = img
                    
                    # Load model
                    model = model_loader.load_model(
                        st.session_state.modality, 
                        selected_model_key
                    )
                    
                    # Make prediction
                    predictions = model.predict(preprocessed)
                    st.session_state.predictions = predictions
                    
                    # Handle different output formats
                    class_names = model_info['classes']
                    
                    # Check if model outputs single value (sigmoid) or two values (softmax)
                    if predictions[0].shape[0] == 1:
                        # Single output (sigmoid): convert to two-class probabilities
                        prob_class1 = float(predictions[0][0])
                        prob_class0 = 1.0 - prob_class1
                        predictions_2class = np.array([[prob_class0, prob_class1]])
                        predicted_class = 1 if prob_class1 > 0.5 else 0
                    else:
                        # Two outputs (softmax): use as-is
                        predictions_2class = predictions
                        predicted_class = np.argmax(predictions[0])
                    
                    # Store processed predictions for XAI methods
                    st.session_state.predictions_2class = predictions_2class
                    st.session_state.predicted_class = predicted_class
                    st.session_state.class_names = class_names
                    
                    st.success("✅ Classification Complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error during classification: {str(e)}")
                    st.info("Note: For full functionality, trained models need to be placed in the models/ directory")
        
        # Display classification results (always shown if available)
        if st.session_state.predictions is not None:
            class_names = st.session_state.get('class_names', model_info['classes'])
            predictions_2class = st.session_state.predictions_2class
            predicted_class = st.session_state.predicted_class
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Predicted Class",
                    class_names[predicted_class],
                    f"{predictions_2class[0][predicted_class]*100:.2f}% confidence"                             
                )
            
            with col2:
                # Show all class probabilities
                st.markdown("**Class Probabilities:**")
                for i, class_name in enumerate(class_names):
                    st.progress(
                        float(predictions_2class[0][i]),
                        text=f"{class_name}: {predictions_2class[0][i]*100:.1f}%"
                    )
            
            # Show preprocessed input
            if st.session_state.modality == 'audio':
                with st.expander("View Spectrogram"):
                    st.image(
                        st.session_state.preprocessed_input,
                        caption="Mel-Spectrogram",
                        use_container_width=True
                    )
        
        # XAI Section
        if st.session_state.predictions is not None:
            st.markdown("---")
            st.markdown("### 4 - Explain Predictions (XAI)")
            
            # Get compatible XAI methods
            compatible_methods = XAICompatibilityChecker.get_compatible_methods(
                st.session_state.modality
            )
            
            xai_display_names = {
                'lime': 'LIME - Local Interpretable Model-agnostic Explanations',
                'gradcam': 'Grad-CAM - Gradient-weighted Class Activation Mapping',
                'shap': 'SHAP - SHapley Additive exPlanations'
            }
            
            selected_xai = st.selectbox(
                "Select XAI Method:",
                options=compatible_methods,
                format_func=lambda x: xai_display_names.get(x, x),
                help="Choose an explainability method compatible with your input"
            )
            
            if st.button("Generate Explanation", type="primary"):
                with st.spinner(f"Generating {selected_xai.upper()} explanation..."):
                    try:
                        # Load model again
                        model = model_loader.load_model(
                            st.session_state.modality,
                            selected_model_key,
                        )
                        
                        class_names = model_info['classes']
                        
                        if selected_xai == 'lime':
                            explainer = LIMEExplainer()
                            # Get model's expected input size
                            target_size = model_info['input_shape'][:2]  # (height, width)
                            result = explainer.explain(
                                st.session_state.preprocessed_input,
                                model,
                                class_names,
                                target_size=target_size
                            )
                            fig = explainer.visualize(result)
                            
                        elif selected_xai == 'gradcam':
                            explainer = GradCAMExplainer(model)
                            # Use the stored predicted class to ensure consistency
                            predicted_class = st.session_state.predicted_class
                            
                            # Preprocess input for Grad-CAM (handles both audio spectrograms and images)
                            from PIL import Image
                            if isinstance(st.session_state.preprocessed_input, Image.Image):
                                # Both audio spectrograms and images are PIL Images
                                if st.session_state.modality == 'image':
                                    preprocessor = ImagePreprocessor()
                                else:  # audio
                                    preprocessor = AudioPreprocessor()
                                
                                # Convert PIL image to array and preprocess
                                img_for_gradcam = preprocessor.preprocess_for_model(st.session_state.preprocessed_input)
                                # Remove batch dimension for Grad-CAM (it will add it back)
                                img_for_gradcam = img_for_gradcam[0]
                            else:
                                # Already preprocessed array
                                img_for_gradcam = st.session_state.preprocessed_input
                            
                            result = explainer.explain(
                                img_for_gradcam,
                                class_idx=predicted_class
                            )
                            fig = explainer.visualize(
                                result,
                                class_name=class_names[predicted_class]
                            )
                            
                        elif selected_xai == 'shap':
                            # Get model's expected input size
                            target_size = model_info['input_shape'][:2]  # (height, width)
                            
                            # Create background samples with correct shape
                            # Background should be (num_samples, height, width, channels)
                            input_shape = tuple(model_info['input_shape'])
                            background_samples = []
                            for i in range(10):
                                # Create varied backgrounds from 0% to 90% intensity
                                background = np.ones(input_shape) * (i / 10.0)
                                background_samples.append(background)
                            background_samples = np.array(background_samples)
                            
                            explainer = SHAPExplainer(model, background_samples)
                            result = explainer.explain(
                                st.session_state.preprocessed_input,
                                normalize=True,
                                nsamples=500,
                                target_size=target_size
                            )
                            fig = explainer.visualize(result, class_names)
                        
                        # Store result
                        st.session_state.xai_results[selected_xai] = {
                            'result': result,
                            'figure': fig
                        }
                        
                        # Display
                        st.pyplot(fig)
                        
                        st.success(f"✅ {selected_xai.upper()} explanation generated successfully!")
                        
                        st.markdown("### Understanding Different XAI Methods:")
    
                        st.markdown("""                      
                        - **LIME**: Highlights superpixels/regions that contributed most to the prediction. 
                        Good for understanding local decision boundaries.
                        
                        - **Grad-CAM**: Shows which spatial regions the model focused on using gradient information.
                        Excellent for localization in images/spectrograms.
                        
                        - **SHAP**: Provides pixel-level attribution based on game theory (Shapley values).
                        Offers comprehensive and theoretically grounded explanations.
                        """)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating explanation: {str(e)}")
                        st.exception(e)


def show_comparison_page():
    """XAI comparison visualization page"""
    st.title("🔍 XAI Method Comparison")
    
    if not st.session_state.xai_results:
        st.info("ℹ️ No XAI results available. Please run classification and generate explanations first.")
        st.markdown("Go to the **Classification & XAI** page to get started.")
        return
    
    st.markdown("### Compare Multiple XAI Explanations")
    st.markdown("View and compare different explainability methods side-by-side")
    
    st.markdown("---")
    
    # Display all generated XAI results
    num_results = len(st.session_state.xai_results)
    
    if num_results == 1:
        # Single result - full width
        method_name = list(st.session_state.xai_results.keys())[0]
        st.markdown(f"### {method_name.upper()} Explanation")
        st.pyplot(st.session_state.xai_results[method_name]['figure'])
        
    elif num_results == 2:
        # Two results - side by side
        col1, col2 = st.columns(2)
        methods = list(st.session_state.xai_results.keys())
        
        with col1:
            st.markdown(f"### {methods[0].upper()}")
            st.pyplot(st.session_state.xai_results[methods[0]]['figure'])
        
        with col2:
            st.markdown(f"### {methods[1].upper()}")
            st.pyplot(st.session_state.xai_results[methods[1]]['figure'])
            
    else:
        # Three or more results - grid layout
        methods = list(st.session_state.xai_results.keys())
        
        for i in range(0, len(methods), 2):
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                if i + j < len(methods):
                    method = methods[i + j]
                    with col:
                        st.markdown(f"### {method.upper()}")
                        st.pyplot(st.session_state.xai_results[method]['figure'])
    
    st.markdown("---")
    
    # Comparison insights
    st.markdown("### 📊 Comparison Insights")
    
    st.markdown("""
    **Understanding Different XAI Methods:**
    
    - **LIME**: Highlights superpixels/regions that contributed most to the prediction. 
      Good for understanding local decision boundaries.
      
    - **Grad-CAM**: Shows which spatial regions the model focused on using gradient information.
      Excellent for localization in images/spectrograms.
      
    - **SHAP**: Provides pixel-level attribution based on game theory (Shapley values).
      Offers comprehensive and theoretically grounded explanations.
    """)
    
    if st.button("Clear All Results"):
        st.session_state.xai_results = {}
        st.rerun()


if __name__ == "__main__":
    main()
