"""
XAI method compatibility checker
"""


class XAICompatibilityChecker:
    """Checks which XAI methods are compatible with different input modalities"""
    
    # Compatibility matrix
    COMPATIBILITY = {
        'audio': {
            'lime': True,
            'gradcam': True,
            'shap': True
        },
        'image': {
            'lime': True,
            'gradcam': False,
            'shap': True
        }
    }
    
    @classmethod
    def is_compatible(cls, modality, xai_method):
        """
        Check if an XAI method is compatible with a modality
        
        Args:
            modality: 'audio' or 'image'
            xai_method: 'lime', 'gradcam', or 'shap'
            
        Returns:
            Boolean indicating compatibility
        """
        if modality not in cls.COMPATIBILITY:
            return False
        
        return cls.COMPATIBILITY[modality].get(xai_method, False)
    
    @classmethod
    def get_compatible_methods(cls, modality):
        """
        Get list of compatible XAI methods for a modality
        
        Args:
            modality: 'audio' or 'image'
            
        Returns:
            List of compatible XAI method names
        """
        if modality not in cls.COMPATIBILITY:
            return []
        
        return [
            method 
            for method, compatible in cls.COMPATIBILITY[modality].items() 
            if compatible
        ]
