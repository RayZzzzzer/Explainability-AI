"""
LIME (Local Interpretable Model-agnostic Explanations) implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image


class LIMEExplainer:
    """LIME explainer for both audio spectrograms and images"""
    
    def __init__(self):
        self.explainer = lime_image.LimeImageExplainer()
    
    def explain(self, image, model, class_names, num_samples=1000, 
                num_features=8, positive_only=False):
        """
        Generate LIME explanation for an image/spectrogram
        
        Args:
            image: Input image (PIL Image or numpy array)
            model: Trained model with predict method
            class_names: List of class names
            num_samples: Number of samples for LIME
            num_features: Number of superpixels to show
            positive_only: Show only positive contributions
            
        Returns:
            Dictionary with explanation data and visualization
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            # Force convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
        else:
            img_array = image
        
        # Ensure 3 channels (RGB)
        if len(img_array.shape) == 2:
            # Grayscale - convert to RGB by repeating channels
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            # RGBA - remove alpha channel
            img_array = img_array[:, :, :3]
        elif img_array.shape[-1] != 3:
            # Other formats - try to fix
            raise ValueError(f"Image must have 3 channels (RGB), got shape {img_array.shape}")
        
        # Normalize if needed
        if img_array.max() > 1.0:
            img_normalized = img_array / 255.0
        else:
            img_normalized = img_array
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            img_normalized.astype('float64'),
            model.predict,
            top_labels=len(class_names),
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get prediction
        predictions = model.predict(np.expand_dims(img_normalized, axis=0))
        predicted_class = np.argmax(predictions[0])
        
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=False
        )
        
        return {
            'explanation': explanation,
            'image_with_mask': mark_boundaries(temp, mask),
            'predicted_class': predicted_class,
            'predicted_class_name': class_names[predicted_class],
            'predictions': predictions[0],
            'original_image': img_normalized
        }
    
    def visualize(self, explanation_result, figsize=(12, 5)):
        """
        Create visualization of LIME explanation
        
        Args:
            explanation_result: Result from explain method
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        axes[0].imshow(explanation_result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # LIME explanation
        axes[1].imshow(explanation_result['image_with_mask'])
        axes[1].set_title(f"LIME Explanation\nPredicted: {explanation_result['predicted_class_name']}")
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_comparison(self, original_image, explanation_result, 
                           class_names, figsize=(15, 5)):
        """
        Create detailed comparison visualization
        
        Args:
            original_image: Original input image
            explanation_result: Result from explain method
            class_names: List of class names
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original
        axes[0].imshow(original_image)
        axes[0].set_title('Original Input')
        axes[0].axis('off')
        
        # LIME mask
        axes[1].imshow(explanation_result['image_with_mask'])
        axes[1].set_title('LIME Explanation')
        axes[1].axis('off')
        
        # Prediction probabilities
        predictions = explanation_result['predictions']
        axes[2].barh(class_names, predictions)
        axes[2].set_xlabel('Probability')
        axes[2].set_title('Class Probabilities')
        axes[2].set_xlim([0, 1])
        
        plt.tight_layout()
        return fig
