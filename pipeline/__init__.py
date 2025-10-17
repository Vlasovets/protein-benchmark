"""
Protein Analysis Pipeline
==========================

A modular Python pipeline for protein-based machine learning analysis,
ported from R analysis workflows.

This pipeline provides a comprehensive toolkit for:
- Data loading and preprocessing
- Protein-Wide Association Study (PWAS) analysis
- Feature importance analysis using Random Forest
- Recursive Feature Elimination (RFE)
- Model training and evaluation
- Checkpoint system for long-running analyses

Modules:
--------
- data_loader: Handle data loading and matrix preparation
- pwas_analysis: Perform PWAS with multiple testing correction
- feature_importance: Random Forest feature importance analysis
- feature_selection: RFE and other feature selection methods
- model_training: Train and evaluate ML models
- checkpoint_system: Save/load intermediate results
- utils: Common utility functions

Example Usage:
--------------
```python
from pipeline import (
    DataLoader, PWASAnalyzer, FeatureImportanceAnalyzer,
    FeatureSelector, ModelTrainer, CheckpointSystem
)

# Initialize components
loader = DataLoader("/path/to/data")
pwas = PWASAnalyzer()
checkpoint = CheckpointSystem("/path/to/checkpoints")

# Load data
data = loader.load_matrices(
    protein_train_path="protein_train.csv",
    phenotype_train_path="phenotype_train.csv",
    protein_val_path="protein_val.csv", 
    phenotype_val_path="phenotype_val.csv"
)

# Run PWAS analysis
pwas_results = pwas.perform_pwas(
    data['p_mtx_traintest'],
    data['pheno_train_test'],
    target_column='oa_status'
)

# Save checkpoint
checkpoint.save_checkpoint('pwas_results', pwas_results)
```

Version: 1.0.0
Author: Protein Analysis Pipeline
"""

# Import main classes for easy access
try:
    from .data_loader import DataLoader, create_sample_data
    from .pwas_analysis import PWASAnalyzer, save_pwas_results, load_pwas_results
    from .feature_importance import (
        FeatureImportanceAnalyzer, save_importance_results, load_importance_results
    )
    from .feature_selection import (
        FeatureSelector, save_selection_results, load_selection_results
    )
    from .model_training import ModelTrainer, save_model_results, load_model_results
    from .checkpoint_system import CheckpointSystem
    from .utils import (
        validate_data_compatibility, create_results_summary,
        print_pipeline_summary, timer, calculate_performance_metrics,
        create_feature_matrix_from_selection, get_system_info
    )
    
    __all__ = [
        # Main classes
        'DataLoader',
        'PWASAnalyzer', 
        'FeatureImportanceAnalyzer',
        'FeatureSelector',
        'ModelTrainer',
        'CheckpointSystem',
        
        # Data functions
        'create_sample_data',
        
        # Save/load functions
        'save_pwas_results', 'load_pwas_results',
        'save_importance_results', 'load_importance_results',
        'save_selection_results', 'load_selection_results',
        'save_model_results', 'load_model_results',
        
        # Utility functions
        'validate_data_compatibility',
        'create_results_summary',
        'print_pipeline_summary',
        'timer',
        'calculate_performance_metrics',
        'create_feature_matrix_from_selection',
        'get_system_info'
    ]
    
    # Check for optional dependencies
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        _HAS_PLOTTING = True
    except ImportError:
        _HAS_PLOTTING = False
    
    try:
        import psutil
        _HAS_PSUTIL = True
    except ImportError:
        _HAS_PSUTIL = False
    
    # Add plotting capability info
    if not _HAS_PLOTTING:
        import warnings
        warnings.warn(
            "Matplotlib/Seaborn not available. Plotting functions will be limited.",
            ImportWarning
        )
    
    if not _HAS_PSUTIL:
        import warnings
        warnings.warn(
            "psutil not available. System information will be limited.",
            ImportWarning
        )
    
except ImportError as e:
    # Handle import errors gracefully
    import warnings
    warnings.warn(f"Some pipeline modules could not be imported: {e}")
    
    __all__ = []

# Pipeline version and metadata
__version__ = "1.0.0"
__author__ = "Protein Analysis Pipeline"
__description__ = "Modular Python pipeline for protein-based machine learning analysis"

def get_pipeline_info():
    """Get information about the pipeline and its capabilities."""
    
    info = {
        'version': __version__,
        'description': __description__,
        'modules_available': len(__all__) > 0,
        'plotting_available': _HAS_PLOTTING if 'HAS_PLOTTING' in globals() else False,
        'system_info_available': _HAS_PSUTIL if '_HAS_PSUTIL' in globals() else False,
        'available_modules': __all__
    }
    
    return info