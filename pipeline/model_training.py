"""
Model Training Module
=====================

Handles training and evaluation of machine learning models
for protein-based classification tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# Import ROSE sampling and imbalanced-learn pipeline
try:
    from imblearn.over_sampling import ROSE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: imbalanced-learn not installed. ROSE sampling will not be available.")
    print("   Install with: pip install imbalanced-learn")
    IMBLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Trains and evaluates machine learning models for classification.
    
    Reproduces R model training functionality with Python, including ROSE sampling.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.training_results = {}
        self.scaler = None
        
    def _get_sampling_strategy(self, sampling_strategy: str):
        """Get the appropriate sampling strategy object"""
        if sampling_strategy is None or sampling_strategy == 'none':
            return None
        elif sampling_strategy == 'rose':
            if not IMBLEARN_AVAILABLE:
                raise ValueError("ROSE sampling requires imbalanced-learn. Install with: pip install imbalanced-learn")
            return ROSE(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")
    
    def _create_model_pipeline(self, model_type: str, sampling_strategy: str = None):
        """Create a pipeline with optional sampling"""
        # Get the base model
        base_model = self._get_default_model(model_type)
        
        # Create pipeline with or without sampling
        sampler = self._get_sampling_strategy(sampling_strategy)
        if sampler is not None and IMBLEARN_AVAILABLE:
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('classifier', base_model)
            ])
        else:
            pipeline = base_model
            
        return pipeline
    
    def train_model(self,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: pd.DataFrame = None,
                   y_val: pd.Series = None,
                   model_type: str = 'random_forest',
                   hyperparameter_tuning: bool = False,
                   cv_folds: int = 10,
                   cv_repeats: int = 1,
                   scale_features: bool = False,
                   sampling_strategy: str = None) -> Dict[str, Any]:
        """
        Train a machine learning model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features matrix
        y_train : pd.Series
            Training target variable
        X_val : pd.DataFrame, optional
            Validation features matrix
        y_val : pd.Series, optional
            Validation target variable
        model_type : str
            Type of model: 'random_forest', 'logistic_regression', 'svm', 'gradient_boosting', 'xgboost'
        hyperparameter_tuning : bool
            Whether to perform hyperparameter tuning
        cv_folds : int
            Number of cross-validation folds
        cv_repeats : int
            Number of cross-validation repeats (for RepeatedStratifiedKFold)
        scale_features : bool
            Whether to scale features (recommended for some models)
        sampling_strategy : str
            Sampling strategy ('rose', 'none', or None)
            
        Returns:
        --------
        dict
            Dictionary containing training results and model performance
        """
        
        print(f"🤖 Training {model_type} model...")
        print(f"  📊 Training: {X_train.shape[0]} samples × {X_train.shape[1]} features")
        if X_val is not None:
            print(f"  🧪 Validation: {X_val.shape[0]} samples")
        if sampling_strategy:
            print(f"  🌹 Sampling strategy: {sampling_strategy}")
        
        # Validate inputs
        self._validate_inputs(X_train, y_train, X_val, y_val)
        
        # Feature scaling if requested
        if scale_features:
            X_train_scaled, X_val_scaled = self._scale_features(X_train, X_val)
        else:
            X_train_scaled, X_val_scaled = X_train.copy(), X_val.copy() if X_val is not None else None
        
        # Initialize model with pipeline
        if hyperparameter_tuning:
            print(f"  🔧 Performing hyperparameter tuning...")
            model = self._get_model_with_tuning(model_type, X_train_scaled, y_train, cv_folds, cv_repeats, sampling_strategy)
        else:
            model = self._create_model_pipeline(model_type, sampling_strategy)
        
        # Train the model
        print(f"  🔄 Training model...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        print(f"  📈 Evaluating performance...")
        performance = self._evaluate_model(
            model, X_train_scaled, y_train, X_val_scaled, y_val, cv_folds, cv_repeats
        )
        
        # Store model and results
        self.models[model_type] = model
        
        results = {
            'model': model,
            'model_type': model_type,
            'performance': performance,
            'features': list(X_train.columns),
            'training_params': {
                'hyperparameter_tuning': hyperparameter_tuning,
                'cv_folds': cv_folds,
                'cv_repeats': cv_repeats,
                'scale_features': scale_features,
                'sampling_strategy': sampling_strategy,
                'n_features': X_train.shape[1],
                'n_train_samples': X_train.shape[0],
                'n_val_samples': X_val.shape[0] if X_val is not None else None
            },
            'data_info': {
                'feature_names': list(X_train.columns),
                'target_distribution_train': y_train.value_counts().to_dict(),
                'target_distribution_val': y_val.value_counts().to_dict() if y_val is not None else None
            }
        }
        
        # Add hyperparameter info if tuning was performed
        if hyperparameter_tuning and hasattr(model, 'best_params_'):
            results['best_hyperparameters'] = model.best_params_
            results['cv_results'] = model.cv_results_
        
        self.training_results[model_type] = results
        
        # Print summary
        self._print_training_summary(results)
        
        return results
    
    def compare_models(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_val: pd.DataFrame = None,
                      y_val: pd.Series = None,
                      model_types: List[str] = None,
                      cv_folds: int = 10,
                      cv_repeats: int = 1,
                      sampling_strategy: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on the same dataset.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features matrix
        y_train : pd.Series
            Training target variable
        X_val : pd.DataFrame, optional
            Validation features matrix
        y_val : pd.Series, optional
            Validation target variable
        model_types : List[str], optional
            List of model types to compare
        cv_folds : int
            Number of cross-validation folds
        cv_repeats : int
            Number of cross-validation repeats
        sampling_strategy : str
            Sampling strategy ('rose', 'none', or None)
            
        Returns:
        --------
        dict
            Dictionary with results for each model type
        """
        
        if model_types is None:
            model_types = ['random_forest', 'logistic_regression', 'gradient_boosting']
        
        print(f"🏆 Comparing {len(model_types)} models...")
        if sampling_strategy:
            print(f"🌹 Using sampling strategy: {sampling_strategy}")
        
        comparison_results = {}
        
        for model_type in model_types:
            print(f"\n  🤖 Training {model_type}...")
            
            try:
                result = self.train_model(
                    X_train, y_train, X_val, y_val,
                    model_type=model_type,
                    cv_folds=cv_folds,
                    cv_repeats=cv_repeats,
                    sampling_strategy=sampling_strategy,
                    hyperparameter_tuning=False  # Skip tuning for comparison
                )
                comparison_results[model_type] = result
                
            except Exception as e:
                warnings.warn(f"Failed to train {model_type}: {str(e)}")
        
        # Print comparison summary
        self._print_model_comparison(comparison_results)
        
        return comparison_results
    
    def predict(self, 
                X: pd.DataFrame,
                model_type: str = None,
                return_probabilities: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using trained model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features matrix for prediction
        model_type : str, optional
            Type of model to use (if None, use the most recent)
        return_probabilities : bool
            Whether to return prediction probabilities
            
        Returns:
        --------
        np.ndarray or tuple
            Predictions (and probabilities if requested)
        """
        
        if not self.models:
            raise ValueError("No trained models available. Train a model first.")
        
        if model_type is None:
            model_type = list(self.models.keys())[-1]  # Use most recent
        
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_type]
        
        # Scale features if scaler was used during training
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
            return predictions, probabilities
        
        return predictions
    
    def _validate_inputs(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Validate input data."""
        
        if X_train.shape[0] != len(y_train):
            raise ValueError("Training features and target must have same number of samples")
        
        if X_val is not None and y_val is not None:
            if X_val.shape[0] != len(y_val):
                raise ValueError("Validation features and target must have same number of samples")
            
            if X_train.shape[1] != X_val.shape[1]:
                raise ValueError("Training and validation features must have same number of features")
        
        if len(y_train.unique()) < 2:
            raise ValueError("Target variable must have at least 2 classes")
    
    def _scale_features(self, X_train: pd.DataFrame, 
                       X_val: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler."""
        
        self.scaler = StandardScaler()
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
        
        return X_train_scaled, X_val_scaled
    
    def _get_default_model(self, model_type: str):
        """Get default model configuration."""
        
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=500,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced',
                oob_score=True
            )
        
        elif model_type == 'logistic_regression':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        
        elif model_type == 'svm':
            return SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            )
        
        elif model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0,
                n_jobs=-1,
                use_label_encoder=False
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_model_with_tuning(self, model_type: str, X_train: pd.DataFrame,
                              y_train: pd.Series, cv_folds: int, cv_repeats: int = 1,
                              sampling_strategy: str = None):
        """Get model with hyperparameter tuning."""
        
        # Create pipeline with sampling
        base_pipeline = self._create_model_pipeline(model_type, sampling_strategy)
        param_grids = self._get_r_style_param_grids(model_type, sampling_strategy)
        
        # Set up cross-validation
        if cv_repeats > 1:
            cv = RepeatedStratifiedKFold(
                n_splits=cv_folds, 
                n_repeats=cv_repeats,
                random_state=self.random_state
            )
        else:
            cv = StratifiedKFold(
                n_splits=cv_folds, 
                shuffle=True, 
                random_state=self.random_state
            )
        
        grid_search = GridSearchCV(
            base_pipeline,
            param_grids,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search
    
    def _get_param_grids(self, model_type: str) -> Dict[str, List]:
        """Get parameter grids for hyperparameter tuning (original version)."""
        return self._get_r_style_param_grids(model_type, None)
    
    def _get_r_style_param_grids(self, model_type: str, sampling_strategy: str = None) -> Dict[str, List]:
        """Get hyperparameter grids that match the R script methodology"""
        
        # Adjust parameter names if using pipeline with sampling
        if sampling_strategy and IMBLEARN_AVAILABLE:
            prefix = 'classifier__'
        else:
            prefix = ''
        
        if model_type == 'random_forest':
            # Matching R: mtry around sqrt(features), min.node.size, etc.
            return {
                f'{prefix}n_estimators': [100, 300, 500],
                f'{prefix}max_depth': [5, 10, 15, None],
                f'{prefix}min_samples_split': [2, 5, 10],
                f'{prefix}min_samples_leaf': [1, 2, 4],
                f'{prefix}max_features': ['sqrt', 'log2', 0.6, 0.8]  # mtry equivalents
            }
            
        elif model_type == 'logistic_regression':
            # Matching R GLMNET: alpha and lambda grid
            return {
                f'{prefix}C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # inverse of lambda
                f'{prefix}penalty': ['l1', 'l2', 'elasticnet'],
                f'{prefix}l1_ratio': [0.1, 0.5, 0.7, 0.9],  # alpha in GLMNET
                f'{prefix}solver': ['saga']  # Required for elasticnet
            }
            
        elif model_type == 'gradient_boosting':
            return {
                f'{prefix}n_estimators': [50, 100, 200],
                f'{prefix}learning_rate': [0.01, 0.1, 0.2],
                f'{prefix}max_depth': [3, 5, 7]
            }
        
        elif model_type == 'svm':
            return {
                f'{prefix}C': [0.1, 1, 10],
                f'{prefix}gamma': ['scale', 'auto', 0.001, 0.01],
                f'{prefix}kernel': ['rbf', 'linear']
            }
        
        elif model_type == 'xgboost':
            # Matching R XGBoost grid from the script
            return {
                f'{prefix}n_estimators': [150, 300, 500],
                f'{prefix}max_depth': [1, 3, 5, 7, 10],
                f'{prefix}learning_rate': [0.03, 0.05, 0.1],
                f'{prefix}gamma': [0, 0.1, 0.5],
                f'{prefix}colsample_bytree': [0.8, 1.0],
                f'{prefix}min_child_weight': [1, 3],
                f'{prefix}subsample': [0.6, 0.8, 1.0]
            }
        
        else:
            return {}
    
    def _evaluate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame = None, y_val: pd.Series = None,
                       cv_folds: int = 10, cv_repeats: int = 1) -> Dict[str, Any]:
        """Evaluate model performance."""
        
        performance = {}
        
        # Training performance
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        
        performance['train'] = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'auc': roc_auc_score(y_train, y_train_proba) if y_train_proba is not None else None
        }
        
        # Cross-validation performance
        if cv_repeats > 1:
            cv = RepeatedStratifiedKFold(
                n_splits=cv_folds, 
                n_repeats=cv_repeats,
                random_state=self.random_state
            )
        else:
            cv = StratifiedKFold(
                n_splits=cv_folds, 
                shuffle=True, 
                random_state=self.random_state
            )
        
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        performance['cross_validation'] = {
            'auc_mean': cv_scores.mean(),
            'auc_std': cv_scores.std(),
            'auc_scores': cv_scores.tolist()
        }
        
        # Validation performance (if validation set provided)
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            performance['validation'] = {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'auc': roc_auc_score(y_val, y_val_proba) if y_val_proba is not None else None,
                'classification_report': classification_report(y_val, y_val_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_val, y_val_pred).tolist()
            }
        
        # OOB score for Random Forest (handle pipeline case)
        if hasattr(model, 'oob_score_'):
            performance['oob_score'] = model.oob_score_
        elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            if hasattr(model.named_steps['classifier'], 'oob_score_'):
                performance['oob_score'] = model.named_steps['classifier'].oob_score_
        
        return performance
    
    def _print_training_summary(self, results: Dict[str, Any]):
        """Print summary of training results."""
        
        model_type = results['model_type']
        performance = results['performance']
        params = results['training_params']
        
        print(f"\n🤖 MODEL TRAINING SUMMARY: {model_type.upper()}")
        print(f"  📊 Features: {params['n_features']}")
        print(f"  👥 Training samples: {params['n_train_samples']}")
        
        if params['n_val_samples']:
            print(f"  🧪 Validation samples: {params['n_val_samples']}")
        
        if params.get('sampling_strategy'):
            print(f"  🌹 Sampling strategy: {params['sampling_strategy']}")
        
        # Performance metrics
        if 'train' in performance:
            train_perf = performance['train']
            print(f"  🎯 Training AUC: {train_perf['auc']:.4f}" if train_perf['auc'] else "")
            print(f"  🎯 Training Accuracy: {train_perf['accuracy']:.4f}")
        
        if 'cross_validation' in performance:
            cv_perf = performance['cross_validation']
            print(f"  📊 CV AUC: {cv_perf['auc_mean']:.4f} ± {cv_perf['auc_std']:.4f}")
        
        if 'validation' in performance:
            val_perf = performance['validation']
            print(f"  🧪 Validation AUC: {val_perf['auc']:.4f}" if val_perf['auc'] else "")
            print(f"  🧪 Validation Accuracy: {val_perf['accuracy']:.4f}")
        
        if 'oob_score' in performance:
            print(f"  🎲 OOB Score: {performance['oob_score']:.4f}")
        
        # Hyperparameters if tuning was performed
        if 'best_hyperparameters' in results:
            print(f"  🔧 Best hyperparameters: {results['best_hyperparameters']}")
    
    def _print_model_comparison(self, comparison_results: Dict[str, Dict[str, Any]]):
        """Print comparison of multiple models."""
        
        print(f"\n🏆 MODEL COMPARISON SUMMARY")
        print("  " + "="*80)
        
        # Create comparison table
        models_data = []
        for model_type, result in comparison_results.items():
            perf = result['performance']
            
            row = {
                'Model': model_type,
                'Train AUC': perf.get('train', {}).get('auc', 'N/A'),
                'CV AUC': perf.get('cross_validation', {}).get('auc_mean', 'N/A'),
                'Val AUC': perf.get('validation', {}).get('auc', 'N/A') if 'validation' in perf else 'N/A'
            }
            models_data.append(row)
        
        # Print table
        print(f"  {'Model':<20} {'Train AUC':<12} {'CV AUC':<12} {'Val AUC':<12}")
        print("  " + "-"*60)
        
        for row in models_data:
            train_auc = f"{row['Train AUC']:.4f}" if isinstance(row['Train AUC'], (int, float)) else str(row['Train AUC'])
            cv_auc = f"{row['CV AUC']:.4f}" if isinstance(row['CV AUV'], (int, float)) else str(row['CV AUC'])
            val_auc = f"{row['Val AUC']:.4f}" if isinstance(row['Val AUC'], (int, float)) else str(row['Val AUC'])
            
            print(f"  {row['Model']:<20} {train_auc:<12} {cv_auc:<12} {val_auc:<12}")


def save_model_results(model_results: Dict[str, Any], output_path: str):
    """
    Save model training results to file.
    
    Parameters:
    -----------
    model_results : dict
        Model training results to save
    output_path : str
        Output file path
    """
    import pickle
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full results (without the actual model object for size reasons)
    results_to_save = model_results.copy()
    if 'model' in results_to_save:
        del results_to_save['model']  # Remove model object to reduce file size
    
    with open(output_path, 'wb') as f:
        pickle.dump(results_to_save, f)
    
    print(f"💾 Model results saved to: {output_path}")
    
    # Save model separately
    if 'model' in model_results:
        model_path = output_path.with_suffix('.model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_results['model'], f)
        print(f"🤖 Model object saved to: {model_path}")


def load_model_results(results_path: str) -> Dict[str, Any]:
    """
    Load model training results from file.
    
    Parameters:
    -----------
    results_path : str
        Path to saved model results
        
    Returns:
    --------
    dict
        Loaded model results
    """
    import pickle
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Try to load model object if available
    model_path = Path(results_path).with_suffix('.model.pkl')
    if model_path.exists():
        with open(model_path, 'rb') as f:
            results['model'] = pickle.load(f)
    
    return results