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
        # Convert to absolute path if relative
        if not os.path.isabs(models_dir):
            # Get the directory of the current file (utils/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the project root
            project_root = os.path.dirname(current_dir)
            # Join with models_dir
            self.models_dir = os.path.join(project_root, models_dir)
        else:
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
                    # Skip if a .h5 file with the same name exists (prefer .h5)
                    h5_path = os.path.join(modality_dir, f"{item}.h5")
                    if os.path.exists(h5_path):
                        print(f"Skipping SavedModel '{item}' because .h5 file exists")
                        continue
                    
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
        
        # Try .h5 format first (priority over SavedModel for better compatibility)
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            
            # Try multiple loading strategies for compatibility
            try:
                # Strategy 1: Try with safe_mode=False for legacy models
                import inspect
                sig = inspect.signature(keras.models.load_model)
                if 'safe_mode' in sig.parameters:
                    model = keras.models.load_model(model_path, compile=False, safe_mode=False)
                else:
                    model = keras.models.load_model(model_path, compile=False)
                print("Model loaded successfully")
            except Exception as e1:
                print(f"Standard loading failed: {e1}")
                # Strategy 2: Try with tf.keras directly (better backward compatibility)
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path, compile=False)
                    print("Model loaded with tf.keras")
                except Exception as e2:
                    print(f"tf.keras loading failed: {e2}")
                    raise e1  # Raise original error
            
            # Get input and output shapes
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            
            return model
        
        # Try SavedModel format (only if .h5 doesn't exist)
        if os.path.exists(savedmodel_path) and os.path.isdir(savedmodel_path):
            try:
                print(f"Loading model from {savedmodel_path}")
                # Use tf_keras for legacy SavedModel compatibility
                model = keras.models.load_model(savedmodel_path)
                return model
            except Exception as e:
                print(f"Error loading SavedModel: {e}")
                raise
        
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
