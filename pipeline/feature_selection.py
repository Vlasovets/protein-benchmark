"""
Feature Selection Module
========================

Implements Recursive Feature Elimination (RFE) and other
feature selection methods for the protein analysis pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
from pathlib import Path

class FeatureSelector:
    """
    Performs feature selection using RFE and other methods.
    
    Reproduces R RFE functionality with Python.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize feature selector.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.selector = None
        self.selection_results = None
        
    def run_rfe(self,
                X_train: pd.DataFrame,
                y_train: pd.Series,
                estimator_type: str = 'random_forest',
                n_features_to_select: int = None,
                step: int = 1,
                cv_folds: int = 5,
                scoring: str = 'roc_auc',
                use_cv: bool = True) -> Dict[str, Any]:
        """
        Run Recursive Feature Elimination.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features matrix
        y_train : pd.Series
            Training target variable
        estimator_type : str
            Type of estimator: 'random_forest' or 'logistic_regression'
        n_features_to_select : int, optional
            Number of features to select (if None, use RFECV)
        step : int
            Number of features to remove at each iteration
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring metric for cross-validation
        use_cv : bool
            Whether to use cross-validation for feature selection
            
        Returns:
        --------
        dict
            Dictionary containing RFE results and selected features
        """
        
        print(f"🔧 Starting Recursive Feature Elimination...")
        print(f"  📊 Initial features: {X_train.shape[1]}")
        print(f"  🎯 Target features: {n_features_to_select if n_features_to_select else 'auto (CV)'}")
        print(f"  🏗️  Estimator: {estimator_type}")
        
        # Validate inputs
        self._validate_inputs(X_train, y_train)
        
        # Initialize estimator
        estimator = self._get_estimator(estimator_type)
        
        # Set up RFE or RFECV
        if use_cv and n_features_to_select is None:
            # Use RFECV to automatically determine optimal number of features
            print(f"  🔄 Running RFECV with {cv_folds}-fold CV...")
            
            self.selector = RFECV(
                estimator=estimator,
                step=step,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                n_jobs=-1
            )
        else:
            # Use RFE with specified number of features
            if n_features_to_select is None:
                n_features_to_select = min(20, X_train.shape[1] // 2)  # Default fallback
            
            print(f"  🔄 Running RFE for {n_features_to_select} features...")
            
            self.selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                step=step
            )
        
        # Fit the selector
        self.selector.fit(X_train, y_train)
        
        # Get selected features
        selected_features = X_train.columns[self.selector.support_].tolist()
        feature_rankings = self.selector.ranking_
        
        # Create feature selection DataFrame
        feature_df = pd.DataFrame({
            'feature': X_train.columns,
            'selected': self.selector.support_,
            'ranking': feature_rankings
        }).sort_values('ranking').reset_index(drop=True)
        
        # Evaluate performance
        print(f"  📈 Evaluating feature selection performance...")
        
        X_selected = X_train.iloc[:, self.selector.support_]
        performance = self._evaluate_feature_selection(X_selected, y_train, cv_folds)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(self.selector.estimator_, 'feature_importances_'):
            importance_values = self.selector.estimator_.feature_importances_
            if hasattr(self.selector, 'support_'):
                # For RFE, importance is only for selected features
                # Create full importance array with zeros for unselected features
                full_importance = np.zeros(len(X_train.columns))
                full_importance[self.selector.support_] = importance_values
                
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': full_importance,
                    'selected': self.selector.support_
                }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Package results
        results = {
            'selected_features': selected_features,
            'feature_rankings': feature_df,
            'feature_importance': feature_importance,
            'performance': performance,
            'selection_params': {
                'estimator_type': estimator_type,
                'n_features_selected': len(selected_features),
                'original_n_features': X_train.shape[1],
                'step': step,
                'cv_folds': cv_folds,
                'scoring': scoring,
                'use_cv': use_cv
            },
            'data_info': {
                'n_samples': X_train.shape[0],
                'target_distribution': y_train.value_counts().to_dict()
            }
        }
        
        # Add RFECV-specific results if applicable
        if hasattr(self.selector, 'cv_results_'):
            results['cv_results'] = {
                'scores': self.selector.cv_results_,
                'optimal_n_features': self.selector.n_features_
            }
        
        # Store results
        self.selection_results = results
        
        # Print summary
        self._print_rfe_summary(results)
        
        return results
    
    def get_selected_features(self) -> List[str]:
        """
        Get the list of selected features.
        
        Returns:
        --------
        List[str]
            List of selected feature names
        """
        
        if self.selection_results is None:
            raise ValueError("No feature selection results available. Run run_rfe() first.")
        
        return self.selection_results['selected_features']
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature matrix to include only selected features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix to transform
            
        Returns:
        --------
        pd.DataFrame
            Transformed feature matrix with only selected features
        """
        
        if self.selector is None:
            raise ValueError("No feature selector available. Run run_rfe() first.")
        
        selected_features = self.get_selected_features()
        return X[selected_features].copy()
    
    def run_multiple_selection_methods(self,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     methods: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple feature selection methods for comparison.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features matrix
        y_train : pd.Series
            Training target variable
        methods : List[str], optional
            List of methods to run
            
        Returns:
        --------
        dict
            Dictionary with results from each method
        """
        
        if methods is None:
            methods = ['rfe_rf', 'rfe_lr', 'univariate', 'variance_threshold']
        
        print(f"🔍 Comparing {len(methods)} feature selection methods...")
        
        results = {}
        
        for method in methods:
            print(f"\n  🔄 Running {method}...")
            
            try:
                if method == 'rfe_rf':
                    result = self.run_rfe(X_train, y_train, estimator_type='random_forest')
                    results[method] = result
                    
                elif method == 'rfe_lr':
                    result = self.run_rfe(X_train, y_train, estimator_type='logistic_regression')
                    results[method] = result
                    
                elif method == 'univariate':
                    result = self._run_univariate_selection(X_train, y_train)
                    results[method] = result
                    
                elif method == 'variance_threshold':
                    result = self._run_variance_threshold_selection(X_train, y_train)
                    results[method] = result
                    
                else:
                    warnings.warn(f"Unknown method: {method}")
                    
            except Exception as e:
                warnings.warn(f"Failed to run {method}: {str(e)}")
        
        print(f"\n✅ Completed {len(results)} feature selection methods")
        
        return results
    
    def _get_estimator(self, estimator_type: str):
        """Get estimator for feature selection."""
        
        if estimator_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        
        elif estimator_type == 'logistic_regression':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
    
    def _evaluate_feature_selection(self, 
                                   X_selected: pd.DataFrame,
                                   y_train: pd.Series,
                                   cv_folds: int) -> Dict[str, float]:
        """Evaluate performance of selected features."""
        
        # Quick random forest evaluation
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            rf, X_selected, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Training score
        rf.fit(X_selected, y_train)
        y_pred_proba = rf.predict_proba(X_selected)[:, 1]
        train_auc = roc_auc_score(y_train, y_pred_proba)
        
        return {
            'train_auc': train_auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def _run_univariate_selection(self, 
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 k: int = 20) -> Dict[str, Any]:
        """Run univariate feature selection."""
        
        from sklearn.feature_selection import SelectKBest, f_classif
        
        selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
        selector.fit(X_train, y_train)
        
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        feature_scores = pd.DataFrame({
            'feature': X_train.columns,
            'score': selector.scores_,
            'p_value': selector.pvalues_,
            'selected': selector.get_support()
        }).sort_values('score', ascending=False).reset_index(drop=True)
        
        X_selected = X_train.iloc[:, selector.get_support()]
        performance = self._evaluate_feature_selection(X_selected, y_train, 5)
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'performance': performance,
            'selection_params': {
                'method': 'univariate',
                'k': k,
                'score_func': 'f_classif'
            }
        }
    
    def _run_variance_threshold_selection(self,
                                        X_train: pd.DataFrame,
                                        y_train: pd.Series,
                                        threshold: float = 0.0) -> Dict[str, Any]:
        """Run variance threshold feature selection."""
        
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_train)
        
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        feature_variances = pd.DataFrame({
            'feature': X_train.columns,
            'variance': X_train.var(),
            'selected': selector.get_support()
        }).sort_values('variance', ascending=False).reset_index(drop=True)
        
        X_selected = X_train.iloc[:, selector.get_support()]
        performance = self._evaluate_feature_selection(X_selected, y_train, 5)
        
        return {
            'selected_features': selected_features,
            'feature_variances': feature_variances,
            'performance': performance,
            'selection_params': {
                'method': 'variance_threshold',
                'threshold': threshold
            }
        }
    
    def _validate_inputs(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Validate input data."""
        if X_train.shape[0] != len(y_train):
            raise ValueError("Feature matrix and target variable must have same number of samples")
        
        if len(y_train.unique()) < 2:
            raise ValueError("Target variable must have at least 2 classes")
    
    def _print_rfe_summary(self, results: Dict[str, Any]):
        """Print summary of RFE results."""
        
        params = results['selection_params']
        performance = results['performance']
        
        print(f"\n🔧 RFE FEATURE SELECTION SUMMARY")
        print(f"  📊 Features selected: {params['n_features_selected']}/{params['original_n_features']}")
        print(f"  🎯 Selection ratio: {params['n_features_selected']/params['original_n_features']:.2%}")
        print(f"  🏗️  Estimator: {params['estimator_type']}")
        
        print(f"\n  📈 Performance with selected features:")
        print(f"    Training AUC: {performance['train_auc']:.4f}")
        print(f"    CV AUC: {performance['cv_auc_mean']:.4f} ± {performance['cv_auc_std']:.4f}")
        
        # Print top selected features
        selected_features = results['selected_features']
        print(f"\n  🏆 Selected features:")
        if len(selected_features) <= 20:
            for i, feature in enumerate(selected_features, 1):
                print(f"    {i:2d}. {feature}")
        else:
            for i, feature in enumerate(selected_features[:10], 1):
                print(f"    {i:2d}. {feature}")
            print(f"    ... and {len(selected_features)-10} more")
        
        # Print CV results if available
        if 'cv_results' in results:
            cv_results = results['cv_results']
            print(f"\n  📊 RFECV Results:")
            print(f"    Optimal features: {cv_results['optimal_n_features']}")
            
            # Fix the formatting error by handling different data types
            try:
                scores = cv_results['scores']
                if isinstance(scores, (list, tuple)) and len(scores) > 0:
                    # Convert to numeric if they're strings
                    if isinstance(scores[0], str):
                        numeric_scores = [float(s) for s in scores if s not in ['nan', '', 'None']]
                    else:
                        numeric_scores = [float(s) for s in scores if not pd.isna(s)]
                    
                    if numeric_scores:
                        print(f"    Best CV score: {max(numeric_scores):.4f}")
                        print(f"    Score range: {min(numeric_scores):.4f} - {max(numeric_scores):.4f}")
                    else:
                        print(f"    Best CV score: No valid scores available")
                else:
                    print(f"    Best CV score: No scores available")
            except (ValueError, TypeError, KeyError) as e:
                print(f"    Best CV score: Error processing scores ({e})")


def save_selection_results(selection_results: Dict[str, Any], output_path: str):
    """
    Save feature selection results to file.
    
    Parameters:
    -----------
    selection_results : dict
        Feature selection results to save
    output_path : str
        Output file path
    """
    import pickle
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(selection_results, f)
    
    print(f"💾 Feature selection results saved to: {output_path}")
    
    # Save selected features list as text file
    if 'selected_features' in selection_results:
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            for feature in selection_results['selected_features']:
                f.write(f"{feature}\n")
        print(f"📝 Selected features list saved to: {txt_path}")


def load_selection_results(results_path: str) -> Dict[str, Any]:
    """
    Load feature selection results from file.
    
    Parameters:
    -----------
    results_path : str
        Path to saved selection results
        
    Returns:
    --------
    dict
        Loaded selection results
    """
    import pickle
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results