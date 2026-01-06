"""
SHAP (SHapley Additive exPlanations) implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


class SHAPExplainer:
    """SHAP explainer for deep learning models"""
    
    def __init__(self, model, background_data=None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Keras model
            background_data: Background dataset for SHAP (if None, uses zeros)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
            
        self.model = model
        
        # Initialize SHAP explainer
        if background_data is None:
            # Use a small zero background
            input_shape = model.input_shape[1:]
            background_data = np.zeros((1,) + input_shape)
        
        self.explainer = shap.DeepExplainer(model, background_data)
    
    def explain(self, image, normalize=True):
        """
        Generate SHAP explanation
        
        Args:
            image: Input image (PIL Image or numpy array)
            normalize: Whether to normalize the image
            
        Returns:
            Dictionary with SHAP values and visualization data
        """
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Normalize
        if normalize and img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Add batch dimension if needed
        if len(img_array.shape) == 3:
            img_input = np.expand_dims(img_array, axis=0)
        else:
            img_input = img_array
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(img_input)
        
        # Get prediction
        predictions = self.model.predict(img_input)
        predicted_class = np.argmax(predictions[0])
        
        return {
            'shap_values': shap_values,
            'original_image': img_array,
            'predicted_class': predicted_class,
            'predictions': predictions[0]
        }
    
    def visualize(self, explanation_result, class_names=None, figsize=(12, 5)):
        """
        Create visualization of SHAP explanation
        
        Args:
            explanation_result: Result from explain method
            class_names: List of class names
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        shap_values = explanation_result['shap_values']
        predicted_class = explanation_result['predicted_class']
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        axes[0].imshow(explanation_result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # SHAP values for predicted class
        if isinstance(shap_values, list):
            shap_for_class = shap_values[predicted_class][0]
        else:
            shap_for_class = shap_values[0]
        
        # Sum across channels for visualization
        if len(shap_for_class.shape) == 3:
            shap_viz = np.sum(np.abs(shap_for_class), axis=-1)
        else:
            shap_viz = np.abs(shap_for_class)
        
        # Normalize for visualization
        if shap_viz.max() > 0:
            shap_viz = shap_viz / shap_viz.max()
        
        im = axes[1].imshow(shap_viz, cmap='hot', interpolation='bilinear')
        
        title = 'SHAP Attribution'
        if class_names and predicted_class < len(class_names):
            title += f'\nPredicted: {class_names[predicted_class]}'
        axes[1].set_title(title)
        axes[1].axis('off')
        
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        return fig
    
    def visualize_detailed(self, explanation_result, class_names=None, figsize=(15, 5)):
        """
        Create detailed visualization with multiple views
        
        Args:
            explanation_result: Result from explain method
            class_names: List of class names
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        shap_values = explanation_result['shap_values']
        predicted_class = explanation_result['predicted_class']
        
        # Original image
        axes[0].imshow(explanation_result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # SHAP heatmap
        if isinstance(shap_values, list):
            shap_for_class = shap_values[predicted_class][0]
        else:
            shap_for_class = shap_values[0]
        
        if len(shap_for_class.shape) == 3:
            shap_viz = np.sum(np.abs(shap_for_class), axis=-1)
        else:
            shap_viz = np.abs(shap_for_class)
        
        if shap_viz.max() > 0:
            shap_viz = shap_viz / shap_viz.max()
        
        im = axes[1].imshow(shap_viz, cmap='hot', interpolation='bilinear')
        axes[1].set_title('SHAP Attribution Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Prediction probabilities
        predictions = explanation_result['predictions']
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(predictions))]
        
        axes[2].barh(class_names, predictions)
        axes[2].set_xlabel('Probability')
        axes[2].set_title('Class Probabilities')
        axes[2].set_xlim([0, 1])
        
        plt.tight_layout()
        return fig
