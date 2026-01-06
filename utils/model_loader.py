"""
Model loading and management utilities
"""

import tensorflow as tf
# Use tf_keras for compatibility with SavedModel format
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras
import os


class ModelLoader:
    """Handles loading of different model types for audio and image classification"""
    
    # Default metadata for different modalities
    DEFAULT_METADATA = {
        'audio': {
            'description': 'Audio classification model for deepfake detection',
            'input_shape': (224, 224, 3),
            'classes': ['Real', 'Fake']
        },
        'image': {
            'description': 'Image classification model for medical diagnosis',
            'input_shape': (224, 224, 3),
            'classes': ['Benign', 'Malignant']
        }
    }
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.loaded_models = {}
    
    def get_available_models(self, modality):
        """
        Get available models for a specific modality by scanning the file system
        
        Args:
            modality: 'audio' or 'image'
            
        Returns:
            Dictionary of available models with metadata
        """
        if modality not in ['audio', 'image']:
            raise ValueError(f"Unknown modality: {modality}")
        
        modality_dir = os.path.join(self.models_dir, modality)
        available_models = {}
        
        # Check if directory exists
        if not os.path.exists(modality_dir):
            print(f"Warning: Directory {modality_dir} does not exist")
            return available_models
        
        # Scan for .h5 files
        for filename in os.listdir(modality_dir):
            if filename.endswith('.h5'):
                model_key = filename.replace('.h5', '')
                model_name = self._format_model_name(model_key)
                
                available_models[model_key] = {
                    'name': model_name,
                    'description': self.DEFAULT_METADATA[modality]['description'],
                    'input_shape': self.DEFAULT_METADATA[modality]['input_shape'],
                    'classes': self.DEFAULT_METADATA[modality]['classes']
                }
        
        # Also scan for SavedModel directories (folders with saved_model.pb)
        for item in os.listdir(modality_dir):
            item_path = os.path.join(modality_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a SavedModel
                if os.path.exists(os.path.join(item_path, 'saved_model.pb')):
                    model_name = self._format_model_name(item)
                    
                    available_models[item] = {
                        'name': model_name,
                        'description': self.DEFAULT_METADATA[modality]['description'] + ' (SavedModel)',
                        'input_shape': self.DEFAULT_METADATA[modality]['input_shape'],
                        'classes': self.DEFAULT_METADATA[modality]['classes']
                    }
        
        return available_models
    
    def _format_model_name(self, model_key):
        """
        Format a model key into a readable name
        
        Args:
            model_key: Model filename without extension
            
        Returns:
            Formatted model name
        """
        # Replace underscores with spaces and capitalize
        name = model_key.replace('_', ' ').title()
        return name
    
    def get_model_path(self, modality, model_key):
        """Get the file path for a model"""
        return os.path.join(self.models_dir, modality, f"{model_key}.h5")
    
    def model_exists(self, modality, model_key):
        """Check if a model file exists"""
        path = self.get_model_path(modality, model_key)
        return os.path.exists(path)
    
    def list_available_model_files(self, modality):
        """List all .h5 model files in the modality directory"""
        modality_dir = os.path.join(self.models_dir, modality)
        if not os.path.exists(modality_dir):
            return []
        
        return [
            f.replace('.h5', '') 
            for f in os.listdir(modality_dir) 
            if f.endswith('.h5')
        ]
    
    def load_model(self, modality, model_key):
        """
        Load a model (or create dummy for testing)
        
        Args:
            modality: 'audio' or 'image'
            model_key: Model identifier from registry
            
        Returns:
            Loaded Keras model
        """
        # Check cache
        cache_key = f"{modality}_{model_key}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Try to load from file
        model = self._load_from_file(modality, model_key)
        
        # Cache the model
        self.loaded_models[cache_key] = model
        
        return model
    
    def _load_from_file(self, modality, model_key):
        """Load model from file or raise error"""
        # Try to load real model
        model_path = os.path.join(self.models_dir, modality, f"{model_key}.h5")
        
        # Also check for SavedModel format
        savedmodel_path = os.path.join(self.models_dir, modality, model_key)
        
        # Try .h5 format
        if os.path.exists(model_path):
            try:
                print(f"Loading model from {model_path}")
                # Try loading with compile=False and custom_objects for legacy models
                try:
                    # Custom deserializer for legacy DTypePolicy objects
                    def dtype_policy_deserializer(**kwargs):
                        """Deserializer for legacy DTypePolicy objects"""
                        # Just return a simple dtype string
                        config = kwargs.get('config', kwargs)
                        if isinstance(config, dict) and 'name' in config:
                            return config['name']
                        return 'float32'
                    
                    custom_objects = {
                        'DTypePolicy': dtype_policy_deserializer,
                    }
                    
                    model = keras.models.load_model(
                        model_path, 
                        compile=False,
                        custom_objects=custom_objects
                    )
                    print("Model loaded successfully (without compilation)")
                    
                    # Get input and output shapes
                    print(f"Model input shape: {model.input_shape}")
                    print(f"Model output shape: {model.output_shape}")
                    
                    # Recompile with default optimizer
                    model.compile(
                        optimizer='adam', 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy']
                    )
                    return model
                except Exception as e1:
                    print(f"Error loading with custom_objects: {e1}")
                    # Last resort: try default load
                    model = keras.models.load_model(model_path)
                    return model
            except Exception as e:
                print(f"Error loading .h5 model: {e}")
        
        # Try SavedModel format
        if os.path.exists(savedmodel_path) and os.path.isdir(savedmodel_path):
            try:
                print(f"Loading model from {savedmodel_path}")
                # Use tf_keras for legacy SavedModel compatibility
                model = keras.models.load_model(savedmodel_path)
                return model
            except Exception as e:
                print(f"Error loading SavedModel: {e}")
        
        # If no real model found, raise error
        raise FileNotFoundError(
            f"No model found for '{model_key}' in modality '{modality}'. "
            f"Searched paths:\n"
            f"  - {model_path}\n"
            f"  - {savedmodel_path}\n"
            f"Please ensure the model files are present in the models directory."
        )
    
    def get_model_info(self, model_key, modality):
        """Get metadata about a model"""
        models = self.get_available_models(modality)
        return models.get(model_key, None)
