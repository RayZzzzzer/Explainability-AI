"""
SHAP (SHapley Additive exPlanations) implementation
Falls back to Integrated Gradients when SHAP explainers are not compatible with tf_keras models
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


def integrated_gradients(model, image, baseline=None, steps=50):
    """
    Compute Integrated Gradients attribution map.
    Works with any TensorFlow/Keras model.
    
    Args:
        model: TensorFlow/Keras model
        image: Input image (batch, height, width, channels)
        baseline: Baseline image (default: zeros)
        steps: Number of integration steps
        
    Returns:
        Tuple of (attribution map, predicted class)
    """
    if baseline is None:
        baseline = np.zeros_like(image)
    
    # Convert to tensors
    image_tensor = tf.constant(image, dtype=tf.float32)
    baseline_tensor = tf.constant(baseline, dtype=tf.float32)
    
    # Generate interpolated images
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    interpolated_images = []
    
    for alpha in alphas:
        interpolated = baseline_tensor + alpha * (image_tensor - baseline_tensor)
        interpolated_images.append(interpolated)
    
    interpolated_images = tf.concat(interpolated_images, axis=0)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = model(interpolated_images)
        # Get predicted class for original image
        pred_class = tf.argmax(predictions[-1])
        # Extract score for predicted class
        target_scores = predictions[:, pred_class]
    
    # Get gradients
    gradients = tape.gradient(target_scores, interpolated_images)
    
    # Average gradients
    avg_gradients = tf.reduce_mean(gradients, axis=0, keepdims=True)
    
    # Compute integrated gradients
    integrated_grads = (image_tensor - baseline_tensor) * avg_gradients
    
    return integrated_grads.numpy(), pred_class.numpy()


class SHAPExplainer:
    """
    SHAP explainer for deep learning models.
    Automatically tries GradientExplainer and DeepExplainer, falling back to Integrated Gradients.
    """
    
    def __init__(self, model, background_data=None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Keras/TensorFlow model
            background_data: Background dataset for SHAP (if None, uses zeros)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
            
        self.model = model
        self.use_integrated_gradients = False
        
        # Initialize background data
        if background_data is None:
            input_shape = model.input_shape[1:]
            background_data = np.zeros((1,) + input_shape)
        
        print("Initializing SHAP explainer...")
        
        # Try GradientExplainer (best for CNNs)
        try:
            print("Attempting GradientExplainer...")
            self.explainer = shap.GradientExplainer(model, background_data)
            self.explainer_type = 'gradient'
            print("✓ Using GradientExplainer")
            return
        except Exception as e:
            print(f"GradientExplainer failed: {e}")
        
        # Try DeepExplainer
        try:
            print("Attempting DeepExplainer...")
            self.explainer = shap.DeepExplainer(model, background_data)
            self.explainer_type = 'deep'
            print("✓ Using DeepExplainer")
            return
        except Exception as e:
            print(f"DeepExplainer failed: {e}")
        
        # Fallback to Integrated Gradients
        print("⚠️ SHAP explainers not compatible with this model")
        print("✓ Using Integrated Gradients (gradient-based attribution)")
        self.use_integrated_gradients = True
        self.explainer_type = 'integrated_gradients'
        self.background_data = background_data
    
    def explain(self, image, normalize=True, nsamples=100, target_size=(224, 224)):
        """
        Generate explanation for an image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            normalize: Whether to normalize input to [0, 1]
            nsamples: Number of samples for SHAP (ignored for Integrated Gradients)
            target_size: Model's expected input size (height, width)
            
        Returns:
            Dictionary with 'shap_values', 'original_image', 'predicted_class', 'predictions'
        """
        # Convert to PIL Image for resizing
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Ensure RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize to target size
        pil_image_resized = pil_image.resize(target_size)
        img_array = np.array(pil_image_resized)
        
        # Add batch dimension
        img_input = np.expand_dims(img_array, axis=0)
        
        # Normalize to [0, 1]
        if normalize and img_input.max() > 1.0:
            img_input = img_input.astype('float32') / 255.0
        
        # Create prediction wrapper to handle sigmoid outputs
        def predict_wrapper(x):
            preds = self.model.predict(x, verbose=0)
            # Handle single output (sigmoid) models
            if preds.shape[-1] == 1:
                # Convert to 2-class probabilities
                prob_class1 = preds[:, 0]
                prob_class0 = 1.0 - prob_class1
                return np.column_stack([prob_class0, prob_class1])
            return preds
        
        # Get predictions using wrapper
        predictions = predict_wrapper(img_input)
        predicted_class = np.argmax(predictions[0])
        
        # Compute attribution
        if self.use_integrated_gradients:
            print("Computing Integrated Gradients attribution...")
            baseline = np.mean(self.background_data, axis=0, keepdims=True)
            integrated_grads, _ = integrated_gradients(
                self.model, img_input, baseline=baseline, steps=50
            )
            shap_values = integrated_grads[0]
            
        else:  # GradientExplainer or DeepExplainer
            print(f"Computing SHAP values with {self.explainer_type}...")
            shap_vals = self.explainer.shap_values(img_input, nsamples=nsamples)
            
            # Extract values for predicted class
            if isinstance(shap_vals, list):
                shap_values = shap_vals[predicted_class][0]
            else:
                shap_values = shap_vals[0]
        
        # Ensure shap_values is 3D (height, width, channels), not 4D
        shap_values = np.squeeze(shap_values)
        
        return {
            'shap_values': shap_values,
            'original_image': img_array,
            'predicted_class': predicted_class,
            'predictions': predictions[0]
        }
    
    def visualize(self, explanation_result, class_names=None, figsize=(12, 5)):
        """
        Create visualization of explanation.
        
        Args:
            explanation_result: Result from explain()
            class_names: List of class names
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        shap_values = explanation_result['shap_values']
        predicted_class = explanation_result['predicted_class']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        axes[0].imshow(explanation_result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attribution heatmap
        # Sum across channels for visualization
        if len(shap_values.shape) == 4:
            # Remove any extra dimensions first (batch dimension)
            shap_values = np.squeeze(shap_values)
        
        if len(shap_values.shape) == 3:
            shap_viz = np.sum(np.abs(shap_values), axis=-1)
        elif len(shap_values.shape) == 2:
            shap_viz = np.abs(shap_values)
        else:
            # Fallback: try to reshape
            shap_viz = np.abs(shap_values.reshape(shap_values.shape[0], shap_values.shape[1], -1))
            if shap_viz.shape[-1] > 1:
                shap_viz = np.sum(shap_viz, axis=-1)
        
        # Normalize
        if shap_viz.max() > 0:
            shap_viz = shap_viz / shap_viz.max()
        
        im = axes[1].imshow(shap_viz, cmap='hot', interpolation='bilinear')
        
        title = 'Attribution Map'
        if class_names and predicted_class < len(class_names):
            title += f'\nPredicted: {class_names[predicted_class]}'
        axes[1].set_title(title)
        axes[1].axis('off')
        
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        return fig

