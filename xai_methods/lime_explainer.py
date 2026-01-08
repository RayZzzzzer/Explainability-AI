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
                num_features=8, positive_only=False, target_size=(224, 224)):
        """
        Generate LIME explanation for an image/spectrogram
        
        Args:
            image: Input image (PIL Image or numpy array)
            model: Trained model with predict method
            class_names: List of class names
            num_samples: Number of samples for LIME
            num_features: Number of superpixels to show
            positive_only: Show only positive contributions
            target_size: Model's expected input size (height, width)
            
        Returns:
            Dictionary with explanation data and visualization
        """
        # Convert to PIL Image for resizing
        if isinstance(image, Image.Image):
            pil_image = image
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        else:
            # Convert numpy to PIL
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image.astype(np.uint8))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        
        # Resize to model's expected size
        pil_image_resized = pil_image.resize(target_size)
        img_array = np.array(pil_image_resized)
        
        # Normalize to [0, 1]
        if img_array.max() > 1.0:
            img_normalized = img_array / 255.0
        else:
            img_normalized = img_array
        
        # Create prediction function that handles both sigmoid and softmax outputs
        # Also ensures all images are at the correct size
        def predict_fn(images):
            # LIME may pass images at different sizes, so resize them
            batch_size = images.shape[0]
            if images.shape[1:3] != target_size:
                resized_batch = np.zeros((batch_size, target_size[0], target_size[1], 3))
                for i in range(batch_size):
                    img_to_resize = (images[i] * 255).astype(np.uint8) if images[i].max() <= 1.0 else images[i].astype(np.uint8)
                    pil_img = Image.fromarray(img_to_resize)
                    pil_img_resized = pil_img.resize(target_size)
                    resized_batch[i] = np.array(pil_img_resized) / 255.0
                images = resized_batch
            
            preds = model.predict(images, verbose=0)
            # Handle single output (sigmoid) models
            if preds.shape[-1] == 1:
                # Convert to 2-class probabilities
                prob_class1 = preds[:, 0]
                prob_class0 = 1.0 - prob_class1
                return np.column_stack([prob_class0, prob_class1])
            return preds
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            img_normalized.astype('float64'),
            predict_fn,
            top_labels=len(class_names),
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get prediction
        predictions = predict_fn(np.expand_dims(img_normalized, axis=0))
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
    
    def visualize(self, explanation_result: dict, figsize: tuple[int, int] = (12, 5)):
        """
        Create visualization of LIME explanation.
        
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
