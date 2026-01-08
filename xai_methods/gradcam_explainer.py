"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
"""

import numpy as np
import tensorflow as tf
# Use tf_keras for compatibility
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from PIL import Image


class GradCAMExplainer:
    """Grad-CAM explainer for CNN-based models"""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM explainer
        
        Args:
            model: Keras model
            layer_name: Name of the convolutional layer to use (if None, finds last conv layer)
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model"""
        # Try to find convolutional layers by class type
        conv_layers = []
        
        for layer in self.model.layers:
            # Check by class name (more reliable than layer name)
            layer_class = layer.__class__.__name__
            if 'Conv' in layer_class:
                conv_layers.append(layer.name)
                print(f"Found conv layer: {layer.name} (type: {layer_class})")
        
        if conv_layers:
            print(f"Using last conv layer: {conv_layers[-1]}")
            return conv_layers[-1]
        
        # Fallback: check by name
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                print(f"Using conv layer by name: {layer.name}")
                return layer.name
        
        # Print all layer names for debugging
        print("Available layers:")
        for layer in self.model.layers:
            print(f"  - {layer.name} (type: {layer.__class__.__name__})")
        
        raise ValueError("No convolutional layer found in model")
    
    def explain(self, image, class_idx=None, preprocess_fn=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Input image (PIL Image or numpy array)
            class_idx: Index of class to explain (if None, uses predicted class)
            preprocess_fn: Optional preprocessing function
            
        Returns:
            Dictionary with heatmap and visualization data
        """
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Ensure correct shape and normalization
        if len(img_array.shape) == 3:
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            img_input = np.expand_dims(img_array, axis=0)
        else:
            img_input = img_array
            
        # Apply preprocessing if provided
        if preprocess_fn is not None:
            img_input = preprocess_fn(img_input)
        
        # Get prediction if class_idx not provided
        if class_idx is None:
            predictions = self.model.predict(img_input)
            class_idx = np.argmax(predictions[0])
        
        # Create gradient model
        try:
            last_conv_layer = self.model.get_layer(self.layer_name)
        except:
            # Fallback: try to find any conv layer
            self.layer_name = self._find_last_conv_layer()
            last_conv_layer = self.model.get_layer(self.layer_name)
        
        # IMPORTANT: Call model first to build the graph if needed
        try:
            # Test if model has been built
            _ = self.model.output
            print("Model already built")
        except (AttributeError, ValueError) as e:
            print(f"Model not built yet ({e}), calling it to build graph...")
            _ = self.model(img_input)
            print("Model built successfully")
        
        # Build a model that returns both the conv layer output and final predictions
        try:
            print(f"Attempting to create grad_model with inputs={self.model.inputs}, conv_layer={self.layer_name}")
            grad_model = keras.models.Model(
                inputs=self.model.inputs,
                outputs=[last_conv_layer.output, self.model.output]
            )
            print("Grad model created successfully!")
        except (AttributeError, ValueError, RuntimeError) as e:
            print(f"Error creating grad model with model.inputs/output: {e}")
            print(f"Model type: {type(self.model)}")
            print(f"Model class: {self.model.__class__.__name__}")
            
            # Try building a new functional model by calling layers directly
            try:
                print("Trying functional API approach...")
                # Get input tensor
                input_tensor = keras.Input(shape=self.model.input_shape[1:])
                
                # Pass through model to get outputs
                x = input_tensor
                for layer in self.model.layers:
                    x = layer(x)
                    if layer.name == self.layer_name:
                        conv_output = x
                
                # Create new functional model
                grad_model = keras.models.Model(
                    inputs=input_tensor,
                    outputs=[conv_output, x]
                )
                print("Functional model created successfully!")
            except Exception as e2:
                print(f"Functional approach also failed: {e2}")
                
                # Last resort: use model directly with eager execution
                try:
                    print("Trying direct eager execution approach...")
                    grad_model = None  # We'll compute gradients differently
                except Exception as e3:
                    print(f"All approaches failed: {e3}")
                    raise RuntimeError(f"Could not create gradient model. Try using a different model or XAI method.")
        
        # Compute gradients
        if grad_model is not None:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_input)
                # Handle single-output models (sigmoid)
                if predictions.shape[-1] == 1:
                    class_output = predictions[:, 0] if class_idx == 1 else 1 - predictions[:, 0]
                else:
                    class_output = predictions[:, class_idx]
        else:
            # Direct computation without intermediate model
            with tf.GradientTape() as tape:
                tape.watch(img_input)
                # Forward pass through layers
                x = img_input
                for layer in self.model.layers:
                    x = layer(x)
                    if layer.name == self.layer_name:
                        conv_outputs = x
                        tape.watch(conv_outputs)
                predictions = x
                
                # Handle single-output models (sigmoid)
                if predictions.shape[-1] == 1:
                    class_output = predictions[:, 0] if class_idx == 1 else 1 - predictions[:, 0]
                else:
                    class_output = predictions[:, class_idx]
        
        # Get gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Compute weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match input image
        original_shape = img_array.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (original_shape[1], original_shape[0]))
        
        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on original image
        if img_array.max() <= 1.0:
            img_display = np.uint8(255 * img_array)
        else:
            img_display = np.uint8(img_array)
            
        superimposed = cv2.addWeighted(
            img_display, 0.6,
            heatmap_colored, 0.4,
            0
        )
        
        return {
            'heatmap': heatmap_resized,
            'heatmap_colored': heatmap_colored,
            'superimposed': superimposed,
            'original_image': img_array,
            'predicted_class': class_idx,
            'layer_name': self.layer_name
        }
    
    def visualize(self, explanation_result, class_name=None, figsize=(15, 5)):
        """
        Create visualization of Grad-CAM explanation
        
        Args:
            explanation_result: Result from explain method
            class_name: Name of predicted class
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(explanation_result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(explanation_result['heatmap_colored'])
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Superimposed
        title = 'Grad-CAM Overlay'
        if class_name:
            title += f'\nPredicted: {class_name}'
        axes[2].imshow(explanation_result['superimposed'])
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
