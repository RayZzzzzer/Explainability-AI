"""
Saliency Maps (Vanilla Gradients) implementation
Visualizes which pixels/features most influence the model's prediction
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple


class SaliencyExplainer:
    """Saliency Maps explainer using vanilla gradients."""
    
    def __init__(self, model):
        """
        Initialize Saliency explainer.
        
        Args:
            model: Keras model to explain
        """
        self.model = model
        
    def explain(self, input_data: np.ndarray, class_idx: int = None) -> np.ndarray:
        """
        Generate saliency map by computing gradients of output w.r.t. input.
        
        Args:
            input_data: Input image/spectrogram (shape: (1, H, W, C) or (H, W, C))
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            Saliency map as numpy array (same spatial dimensions as input)
        """
        # Ensure input has batch dimension
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Convert to tensor
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        
        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.model(input_tensor, training=False)
            
            # Handle different output formats
            if isinstance(predictions, dict):
                predictions = list(predictions.values())[0]
            
            # Handle binary vs multi-class classification
            output_shape = predictions.shape[-1]
            
            if output_shape == 1:
                # Binary classification with single sigmoid output
                # Output is probability of positive class
                if class_idx is None or class_idx == 1:
                    target_score = predictions[:, 0]
                else:
                    # For class 0, use negative of the output
                    # (gradient will point in opposite direction)
                    target_score = -predictions[:, 0]
            else:
                # Multi-class classification with softmax output
                # If no class specified, use predicted class
                if class_idx is None:
                    class_idx = tf.argmax(predictions[0]).numpy()
                
                # Get the score for target class
                target_score = predictions[:, class_idx]
        
        # Compute gradients
        gradients = tape.gradient(target_score, input_tensor)
        
        # Convert to numpy
        gradients = gradients.numpy()[0]  # Remove batch dimension
        
        # Take maximum across color channels for visualization
        # This shows the strongest gradient magnitude regardless of channel
        saliency = np.max(np.abs(gradients), axis=-1)
        
        return saliency
    
    def visualize(
        self,
        input_data: np.ndarray,
        saliency_map: np.ndarray,
        class_idx: int = None,
        modality: str = 'image',
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize saliency map overlay on original input.
        
        Args:
            input_data: Original input (shape: (1, H, W, C) or (H, W, C))
            saliency_map: Saliency map from explain()
            class_idx: Target class index for title
            modality: 'image' or 'audio' for appropriate visualization
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Remove batch dimension if present
        if len(input_data.shape) == 4:
            input_data = input_data[0]
        
        # Normalize saliency map to [0, 1]
        saliency_normalized = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Original input
        if modality == 'audio':
            # For spectrograms, show as is
            axes[0].imshow(input_data.squeeze(), aspect='auto', origin='lower', cmap='viridis')
            axes[0].set_title('Original Spectrogram')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Frequency')
        else:
            # For images
            img_display = input_data
            if img_display.shape[-1] == 1:
                img_display = img_display.squeeze()
                axes[0].imshow(img_display, cmap='gray')
            else:
                # Normalize to [0, 1] for display
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
                axes[0].imshow(img_display)
            axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot 2: Saliency map
        im = axes[1].imshow(saliency_normalized, cmap='hot', aspect='auto')
        axes[1].set_title('Saliency Map\n(Gradient Magnitude)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Plot 3: Overlay
        if modality == 'audio':
            # For audio, show spectrogram with saliency overlay
            axes[2].imshow(input_data.squeeze(), aspect='auto', origin='lower', cmap='viridis', alpha=0.6)
            axes[2].imshow(saliency_normalized, cmap='hot', aspect='auto', origin='lower', alpha=0.4)
            axes[2].set_title('Overlay')
            axes[2].set_xlabel('Time')
            axes[2].set_ylabel('Frequency')
        else:
            # For images, show image with saliency overlay
            img_display = input_data
            if img_display.shape[-1] == 1:
                img_display = img_display.squeeze()
                axes[2].imshow(img_display, cmap='gray', alpha=0.6)
            else:
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
                axes[2].imshow(img_display, alpha=0.6)
            axes[2].imshow(saliency_normalized, cmap='hot', alpha=0.4)
            axes[2].set_title('Saliency Overlay')
        axes[2].axis('off')
        
        # Add overall title
        title = 'Saliency Map Visualization'
        if class_idx is not None:
            title += f' (Class {class_idx})'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
