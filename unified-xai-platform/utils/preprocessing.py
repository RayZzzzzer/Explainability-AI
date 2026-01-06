"""
Preprocessing utilities for audio and image inputs
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os


class AudioPreprocessor:
    """Handles audio file preprocessing and spectrogram generation"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def load_audio(self, audio_path, sr=22050):
        """Load audio file using librosa"""
        y, sr = librosa.load(audio_path, sr=sr)
        return y, sr
    
    def create_spectrogram(self, audio_path, output_path='temp_spectrogram.png'):
        """
        Convert audio to mel-spectrogram image
        
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
    
    def preprocess_for_model(self, spec_image, normalize=True):
        """
        Preprocess spectrogram image for model input
        
        Args:
            spec_image: PIL Image of spectrogram
            normalize: Whether to normalize pixel values to [0, 1]
            
        Returns:
            Preprocessed numpy array
        """
        img_array = img_to_array(spec_image)
        
        if normalize:
            img_array = img_array / 255.0
            
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch


class ImagePreprocessor:
    """Handles medical image preprocessing"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def load_image(self, image_path):
        """Load image using PIL"""
        img = load_img(image_path, target_size=self.target_size)
        return img
    
    def preprocess_for_model(self, image, model_type='default', normalize=True):
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image or numpy array
            model_type: Type of model ('vgg16', 'alexnet', 'densenet', 'default')
            normalize: Whether to normalize
            
        Returns:
            Preprocessed numpy array
        """
        # Convert to array if PIL Image
        if isinstance(image, Image.Image):
            img_array = img_to_array(image)
        else:
            img_array = image
        
        # Normalize if needed
        if normalize:
            img_array = img_array / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Apply model-specific preprocessing
        if model_type.lower() == 'vgg16':
            from tensorflow.keras.applications.vgg16 import preprocess_input
            img_batch = preprocess_input(img_batch * 255.0)  # VGG expects [0, 255]
        elif model_type.lower() in ['alexnet', 'densenet']:
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_batch = (img_batch - mean) / std
            
        return img_batch
    
    def resize_image(self, image, size):
        """Resize image to specified size"""
        if isinstance(image, np.ndarray):
            return cv2.resize(image, size)
        else:
            return image.resize(size)
