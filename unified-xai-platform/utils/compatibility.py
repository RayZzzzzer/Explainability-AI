"""
XAI method compatibility checking
"""


class XAICompatibilityChecker:
    """Determines which XAI methods are compatible with input modality"""
    
    # XAI method compatibility matrix
    COMPATIBILITY_MATRIX = {
        'lime': {
            'audio': True,
            'image': True,
            'description': 'LIME (Local Interpretable Model-agnostic Explanations)',
            'details': 'Works for both audio spectrograms and images'
        },
        'gradcam': {
            'audio': True,
            'image': True,
            'description': 'Grad-CAM (Gradient-weighted Class Activation Mapping)',
            'details': 'Requires convolutional layers, works for CNNs on both modalities'
        },
        'shap': {
            'audio': True,
            'image': True,
            'description': 'SHAP (SHapley Additive exPlanations)',
            'details': 'Model-agnostic, works for both modalities'
        },
        'integrated_gradients': {
            'audio': True,
            'image': True,
            'description': 'Integrated Gradients',
            'details': 'Attribution method for neural networks'
        }
    }
    
    @staticmethod
    def get_compatible_methods(modality):
        """
        Get list of XAI methods compatible with the given modality
        
        Args:
            modality: 'audio' or 'image'
            
        Returns:
            List of compatible XAI method names
        """
        compatible = []
        for method_name, info in XAICompatibilityChecker.COMPATIBILITY_MATRIX.items():
            if info.get(modality.lower(), False):
                compatible.append(method_name)
        return compatible
    
    @staticmethod
    def is_compatible(method, modality):
        """
        Check if a specific XAI method is compatible with modality
        
        Args:
            method: XAI method name
            modality: 'audio' or 'image'
            
        Returns:
            Boolean indicating compatibility
        """
        method_info = XAICompatibilityChecker.COMPATIBILITY_MATRIX.get(method.lower(), {})
        return method_info.get(modality.lower(), False)
    
    @staticmethod
    def get_method_info(method):
        """Get information about a specific XAI method"""
        return XAICompatibilityChecker.COMPATIBILITY_MATRIX.get(method.lower(), {})
    
    @staticmethod
    def get_all_methods():
        """Get list of all available XAI methods"""
        return list(XAICompatibilityChecker.COMPATIBILITY_MATRIX.keys())
    
    @staticmethod
    def filter_methods(methods, modality):
        """
        Filter a list of XAI methods to only compatible ones
        
        Args:
            methods: List of method names to filter
            modality: 'audio' or 'image'
            
        Returns:
            Filtered list of compatible methods
        """
        return [
            method for method in methods 
            if XAICompatibilityChecker.is_compatible(method, modality)
        ]
