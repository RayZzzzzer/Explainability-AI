"""
Model loading and management utilities
"""

import tensorflow as tf
from tensorflow import keras
import os


class ModelLoader:
    """Handles loading of different model types for audio and image classification"""
    
    # Model registry with metadata
    AUDIO_MODELS = {
        'mobilenet': {
            'name': 'MobileNet',
            'description': 'MobileNet for audio deepfake detection',
            'input_shape': (224, 224, 3),
            'classes': ['Real', 'Fake']
        },
        'vgg16_audio': {
            'name': 'VGG16 (Audio)',
            'description': 'VGG16 for audio deepfake detection',
            'input_shape': (224, 224, 3),
            'classes': ['Real', 'Fake']
        },
        'resnet_audio': {
            'name': 'ResNet (Audio)',
            'description': 'ResNet for audio deepfake detection',
            'input_shape': (224, 224, 3),
            'classes': ['Real', 'Fake']
        },
        'custom_cnn_audio': {
            'name': 'Custom CNN (Audio)',
            'description': 'Custom CNN for audio deepfake detection',
            'input_shape': (224, 224, 3),
            'classes': ['Real', 'Fake']
        }
    }
    
    IMAGE_MODELS = {
        'alexnet': {
            'name': 'AlexNet',
            'description': 'AlexNet for lung cancer detection',
            'input_shape': (224, 224, 3),
            'classes': ['Benign', 'Malignant']
        },
        'densenet': {
            'name': 'DenseNet',
            'description': 'DenseNet for lung cancer detection',
            'input_shape': (224, 224, 3),
            'classes': ['Benign', 'Malignant']
        },
        'vgg16_image': {
            'name': 'VGG16 (Image)',
            'description': 'VGG16 for lung cancer detection',
            'input_shape': (224, 224, 3),
            'classes': ['Benign', 'Malignant']
        }
    }
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.loaded_models = {}
    
    def get_available_models(self, modality):
        """
        Get list of available models for a modality
        
        Args:
            modality: 'audio' or 'image'
            
        Returns:
            Dictionary of available models
        """
        if modality.lower() == 'audio':
            return self.AUDIO_MODELS
        elif modality.lower() == 'image':
            return self.IMAGE_MODELS
        else:
            return {}
    
    def load_model(self, model_key, modality, model_path=None):
        """
        Load a model from disk
        
        Args:
            model_key: Key identifying the model
            modality: 'audio' or 'image'
            model_path: Path to model file (if None, uses default path)
            
        Returns:
            Loaded Keras model
        """
        # Check if already loaded
        cache_key = f"{modality}_{model_key}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Determine model path
        if model_path is None:
            model_path = os.path.join(
                self.models_dir, 
                modality, 
                f"{model_key}.h5"
            )
        
        # Load model
        try:
            model = keras.models.load_model(model_path)
            self.loaded_models[cache_key] = model
            return model
        except Exception as e:
            print(f"Error loading model {model_key}: {e}")
            return None
    
    def predict(self, model, preprocessed_input):
        """
        Make prediction using loaded model
        
        Args:
            model: Loaded Keras model
            preprocessed_input: Preprocessed input array
            
        Returns:
            Prediction array
        """
        return model.predict(preprocessed_input)
    
    def get_model_info(self, model_key, modality):
        """Get metadata about a model"""
        models = self.get_available_models(modality)
        return models.get(model_key, None)
    
    def create_dummy_model(self, modality, model_key):
        """
        Create a dummy model for testing when actual models are not available
        
        Args:
            modality: 'audio' or 'image'
            model_key: Model identifier
            
        Returns:
            Simple Keras model
        """
        models = self.get_available_models(modality)
        model_info = models.get(model_key, {})
        input_shape = model_info.get('input_shape', (224, 224, 3))
        num_classes = len(model_info.get('classes', ['Class 0', 'Class 1']))
        
        # Create simple model
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
