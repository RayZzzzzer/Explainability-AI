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
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize ModelLoader with models directory path."""
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
    
    def get_available_models(self, modality: str) -> dict[str, dict]:
        """
        Get available models for a specific modality by scanning the file system.
        
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
    
    def _format_model_name(self, model_key: str) -> str:
        """
        Format a model key into a readable name.
        
        Args:
            model_key: Model filename without extension
            
        Returns:
            Formatted model name
        """
        # Replace underscores with spaces and capitalize
        name = model_key.replace('_', ' ').title()
        return name
    
    def load_model(self, modality: str, model_key: str):
        """
        Load a model from file system.
        
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
            
            # ALWAYS use tf.keras for consistency with GradCAM (which uses tf_keras)
            # This prevents Keras 2/3 compatibility issues
            import tensorflow as tf
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                print("Model loaded with tf.keras")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
            
            # Ensure all layers are trainable for gradient computation (needed for GradCAM)
            if hasattr(model, 'layers'):
                for layer in model.layers:
                    layer.trainable = True

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
