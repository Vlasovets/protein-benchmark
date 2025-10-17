"""
Utilities Module
================

Common utility functions for the protein analysis pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle
from pathlib import Path
import warnings
import time
from datetime import datetime

def validate_data_compatibility(protein_data: pd.DataFrame, 
                               phenotype_data: pd.DataFrame,
                               target_column: str) -> Dict[str, Any]:
    """
    Validate compatibility between protein and phenotype data.
    
    Parameters:
    -----------
    protein_data : pd.DataFrame
        Protein expression matrix
    phenotype_data : pd.DataFrame
        Phenotype data
    target_column : str
        Name of target variable
        
    Returns:
    --------
    dict
        Validation results with warnings and statistics
    """
    
    validation = {
        'compatible': True,
        'warnings': [],
        'statistics': {},
        'recommendations': []
    }
    
    # Check dimensions
    if protein_data.shape[0] != phenotype_data.shape[0]:
        validation['warnings'].append(
            f"Sample count mismatch: proteins={protein_data.shape[0]}, "
            f"phenotypes={phenotype_data.shape[0]}"
        )
        validation['compatible'] = False
    
    # Check target column
    if target_column not in phenotype_data.columns:
        validation['warnings'].append(f"Target column '{target_column}' not found")
        validation['compatible'] = False
    else:
        target_dist = phenotype_data[target_column].value_counts()
        validation['statistics']['target_distribution'] = target_dist.to_dict()
        
        # Check for class imbalance
        if len(target_dist) < 2:
            validation['warnings'].append("Target variable has less than 2 classes")
            validation['compatible'] = False
        else:
            min_class_size = target_dist.min()
            max_class_size = target_dist.max()
            imbalance_ratio = max_class_size / min_class_size
            
            validation['statistics']['imbalance_ratio'] = imbalance_ratio
            
            if imbalance_ratio > 10:
                validation['warnings'].append(
                    f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})"
                )
                validation['recommendations'].append("Consider using class_weight='balanced' in models")
    
    # Check for missing values
    protein_missing = protein_data.isnull().sum().sum()
    phenotype_missing = phenotype_data.isnull().sum().sum()
    
    validation['statistics']['missing_values'] = {
        'proteins': protein_missing,
        'phenotypes': phenotype_missing
    }
    
    if protein_missing > 0:
        validation['warnings'].append(f"Found {protein_missing} missing values in protein data")
        validation['recommendations'].append("Consider imputation or removal of missing values")
    
    if phenotype_missing > 0:
        validation['warnings'].append(f"Found {phenotype_missing} missing values in phenotype data")
    
    # Check data types
    non_numeric_proteins = protein_data.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_proteins) > 0:
        validation['warnings'].append(f"Non-numeric protein columns: {list(non_numeric_proteins)}")
        validation['recommendations'].append("Convert protein data to numeric format")
    
    validation['statistics']['data_types'] = {
        'protein_columns': protein_data.shape[1],
        'numeric_protein_columns': protein_data.select_dtypes(include=[np.number]).shape[1],
        'phenotype_columns': phenotype_data.shape[1]
    }
    
    return validation


def create_results_summary(results_dict: Dict[str, Any], 
                          save_path: str = None) -> Dict[str, Any]:
    """
    Create a comprehensive summary of pipeline results.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results from different pipeline steps
    save_path : str, optional
        Path to save the summary
        
    Returns:
    --------
    dict
        Comprehensive results summary
    """
    
    summary = {
        'pipeline_overview': {
            'steps_completed': list(results_dict.keys()),
            'total_steps': len(results_dict),
            'timestamp': datetime.now().isoformat()
        },
        'data_summary': {},
        'performance_summary': {},
        'feature_summary': {},
        'model_summary': {}
    }
    
    # Extract data information
    for step_name, step_results in results_dict.items():
        if isinstance(step_results, dict) and 'data_info' in step_results:
            data_info = step_results['data_info']
            summary['data_summary'][step_name] = data_info
    
    # Extract performance metrics
    for step_name, step_results in results_dict.items():
        if isinstance(step_results, dict) and 'performance' in step_results:
            performance = step_results['performance']
            summary['performance_summary'][step_name] = performance
    
    # Extract feature information
    for step_name, step_results in results_dict.items():
        if isinstance(step_results, dict):
            if 'selected_features' in step_results:
                summary['feature_summary'][step_name] = {
                    'n_features': len(step_results['selected_features']),
                    'features': step_results['selected_features'][:10]  # Top 10
                }
            elif 'feature_importance' in step_results:
                importance_df = step_results['feature_importance']
                if isinstance(importance_df, pd.DataFrame) and len(importance_df) > 0:
                    summary['feature_summary'][step_name] = {
                        'n_features': len(importance_df),
                        'top_features': importance_df.head(10)['feature'].tolist()
                    }
    
    # Extract model information
    for step_name, step_results in results_dict.items():
        if isinstance(step_results, dict) and 'model_type' in step_results:
            model_info = {
                'model_type': step_results['model_type']
            }
            if 'training_params' in step_results:
                model_info['training_params'] = step_results['training_params']
            summary['model_summary'][step_name] = model_info
    
    # Save summary if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Results summary saved to: {save_path}")
    
    return summary


def print_pipeline_summary(summary: Dict[str, Any]):
    """
    Print a formatted pipeline summary.
    
    Parameters:
    -----------
    summary : dict
        Pipeline summary dictionary
    """
    
    print(f"\n{'='*80}")
    print(f"🧬 PROTEIN ANALYSIS PIPELINE SUMMARY")
    print(f"{'='*80}")
    
    # Overview
    overview = summary['pipeline_overview']
    print(f"\n📋 PIPELINE OVERVIEW")
    print(f"  ✅ Steps completed: {overview['total_steps']}")
    print(f"  🕒 Timestamp: {overview['timestamp']}")
    print(f"  🔧 Steps: {', '.join(overview['steps_completed'])}")
    
    # Data summary
    if summary['data_summary']:
        print(f"\n📊 DATA SUMMARY")
        for step, data_info in summary['data_summary'].items():
            print(f"  {step}:")
            if 'n_samples' in data_info:
                print(f"    👥 Samples: {data_info['n_samples']}")
            if 'n_features' in data_info:
                print(f"    🧬 Features: {data_info['n_features']}")
            if 'target_distribution' in data_info:
                print(f"    🎯 Target distribution: {data_info['target_distribution']}")
    
    # Performance summary
    if summary['performance_summary']:
        print(f"\n📈 PERFORMANCE SUMMARY")
        for step, performance in summary['performance_summary'].items():
            print(f"  {step}:")
            
            if isinstance(performance, dict):
                # Training performance
                if 'train' in performance:
                    train_perf = performance['train']
                    if 'auc' in train_perf and train_perf['auc'] is not None:
                        print(f"    🎯 Training AUC: {train_perf['auc']:.4f}")
                
                # CV performance
                if 'cross_validation' in performance:
                    cv_perf = performance['cross_validation']
                    if 'auc_mean' in cv_perf:
                        print(f"    📊 CV AUC: {cv_perf['auc_mean']:.4f} ± {cv_perf.get('auc_std', 0):.4f}")
                
                # Validation performance
                if 'validation' in performance:
                    val_perf = performance['validation']
                    if 'auc' in val_perf and val_perf['auc'] is not None:
                        print(f"    🧪 Validation AUC: {val_perf['auc']:.4f}")
                
                # CV AUC for other steps
                if 'cv_auc_mean' in performance:
                    print(f"    📊 CV AUC: {performance['cv_auc_mean']:.4f} ± {performance.get('cv_auc_std', 0):.4f}")
    
    # Feature summary
    if summary['feature_summary']:
        print(f"\n🧬 FEATURE SUMMARY")
        for step, feature_info in summary['feature_summary'].items():
            print(f"  {step}:")
            print(f"    📊 Number of features: {feature_info['n_features']}")
            
            if 'features' in feature_info:
                features = feature_info['features']
                if len(features) <= 5:
                    print(f"    🏆 Features: {', '.join(features)}")
                else:
                    print(f"    🏆 Top features: {', '.join(features[:5])}...")
            
            if 'top_features' in feature_info:
                top_features = feature_info['top_features']
                if len(top_features) <= 5:
                    print(f"    🏆 Top features: {', '.join(top_features)}")
                else:
                    print(f"    🏆 Top features: {', '.join(top_features[:5])}...")
    
    # Model summary
    if summary['model_summary']:
        print(f"\n🤖 MODEL SUMMARY")
        for step, model_info in summary['model_summary'].items():
            print(f"  {step}:")
            print(f"    🏗️  Model type: {model_info['model_type']}")
            
            if 'training_params' in model_info:
                params = model_info['training_params']
                if 'n_features' in params:
                    print(f"    📊 Features used: {params['n_features']}")
                if 'hyperparameter_tuning' in params:
                    print(f"    🔧 Hyperparameter tuning: {params['hyperparameter_tuning']}")
    
    print(f"\n{'='*80}")


def save_checkpoint_data(data: Any, checkpoint_path: str, step_name: str):
    """
    Save checkpoint data with metadata.
    
    Parameters:
    -----------
    data : Any
        Data to save
    checkpoint_path : str
        Path to save checkpoint
    step_name : str
        Name of the pipeline step
    """
    
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        'data': data,
        'metadata': {
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'data_type': type(data).__name__
        }
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint_data(checkpoint_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load checkpoint data with metadata.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to checkpoint file
        
    Returns:
    --------
    tuple
        (data, metadata) tuple
    """
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    if isinstance(checkpoint_data, dict) and 'data' in checkpoint_data:
        return checkpoint_data['data'], checkpoint_data.get('metadata', {})
    else:
        # Legacy format - just the data
        return checkpoint_data, {}


def create_feature_matrix_from_selection(
    original_matrix: pd.DataFrame,
    selected_features: List[str],
    validate: bool = True
) -> pd.DataFrame:
    """
    Create feature matrix with only selected features.
    
    Parameters:
    -----------
    original_matrix : pd.DataFrame
        Original feature matrix
    selected_features : List[str]
        List of selected feature names
    validate : bool
        Whether to validate feature names
        
    Returns:
    --------
    pd.DataFrame
        Matrix with only selected features
    """
    
    if validate:
        missing_features = set(selected_features) - set(original_matrix.columns)
        if missing_features:
            warnings.warn(f"Missing features in matrix: {missing_features}")
            selected_features = [f for f in selected_features if f in original_matrix.columns]
    
    return original_matrix[selected_features].copy()


def calculate_performance_metrics(y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray, optional
        Prediction probabilities
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['average_precision'] = average_precision_score(y_true, y_prob)
        except ValueError:
            # Skip if not applicable (e.g., multiclass without proper format)
            pass
    
    return metrics


def timer(func):
    """
    Decorator to time function execution.
    
    Parameters:
    -----------
    func : callable
        Function to time
        
    Returns:
    --------
    callable
        Wrapped function
    """
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"⏱️  {func.__name__} executed in {execution_time:.2f} seconds")
        
        # Add timing info to result if it's a dict
        if isinstance(result, dict):
            result['execution_time'] = execution_time
        
        return result
    
    return wrapper


def safe_division(numerator: float, denominator: float, 
                 default: float = 0.0) -> float:
    """
    Perform safe division with default value for zero denominator.
    
    Parameters:
    -----------
    numerator : float
        Numerator value
    denominator : float
        Denominator value  
    default : float
        Default value if denominator is zero
        
    Returns:
    --------
    float
        Division result or default value
    """
    
    return numerator / denominator if denominator != 0 else default


def format_memory_usage(bytes_used: int) -> str:
    """
    Format memory usage in human-readable format.
    
    Parameters:
    -----------
    bytes_used : int
        Memory usage in bytes
        
    Returns:
    --------
    str
        Formatted memory usage string
    """
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_used < 1024.0:
            return f"{bytes_used:.1f} {unit}"
        bytes_used /= 1024.0
    
    return f"{bytes_used:.1f} TB"


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and logging.
    
    Returns:
    --------
    dict
        System information
    """
    
    import platform
    import psutil
    
    try:
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': format_memory_usage(psutil.virtual_memory().total),
            'memory_available': format_memory_usage(psutil.virtual_memory().available)
        }
    except ImportError:
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': 'unknown',
            'memory_total': 'unknown',
            'memory_available': 'unknown'
        }
    
    return info