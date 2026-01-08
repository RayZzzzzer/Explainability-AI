"""
Preprocessing utilities for audio and image inputs
"""

import numpy as np
import librosa
import librosa.display
from PIL import Image
import matplotlib.pyplot as plt
import io
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class AudioPreprocessor:
    """Handles audio file preprocessing for model input."""
    
    def __init__(self, target_size: tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def create_spectrogram(self, audio_path: str, 
                          output_path: str = 'temp_spectrogram.png') -> Image.Image:
        """
        Convert audio to mel-spectrogram image.
        
        Args:
            audio_path: Path to audio file
            output_path: Path to save spectrogram image
            
        Returns:
            PIL Image of spectrogram
        """
        # Load audio
        y, sr = librosa.load(audio_path)
        
        # Create mel-spectrogram
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        ms = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr, ax=ax)
        
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Load and resize spectrogram image
        spec_image = load_img(output_path, target_size=self.target_size)
        
        return spec_image
    
    def preprocess_for_model(self, spectrogram_image: Image.Image, 
                            target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess spectrogram image for model input.
        
        Args:
            spectrogram_image: PIL Image of spectrogram
            target_size: Target size for model
            
        Returns:
            Preprocessed numpy array ready for model (shape: 1, H, W, 3)
        """
        return ImagePreprocessor._preprocess_image(spectrogram_image, target_size)


class ImagePreprocessor:
    """Handles image preprocessing for model input."""
    
    def preprocess_for_model(self, image: Image.Image, 
                           target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image
            target_size: Target size for model
            
        Returns:
            Preprocessed numpy array ready for model (shape: 1, H, W, 3)
        """
        return self._preprocess_image(image, target_size)
    
    @staticmethod
    def _preprocess_image(image: Image.Image, 
                         target_size: tuple[int, int]) -> np.ndarray:
        """Common preprocessing logic for images."""
        # Resize
        img = image.resize(target_size)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img).astype('float32') / 255.0
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
