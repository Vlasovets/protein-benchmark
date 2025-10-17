"""
Feature Importance Module
=========================

Handles feature importance analysis using Random Forest
and other importance ranking methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
from pathlib import Path

class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using Random Forest and other methods.
    
    Reproduces R RF training and feature importance functionality.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize feature importance analyzer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.rf_model = None
        self.importance_results = None
        
    def analyze_rf_importance(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            n_estimators: int = 500,
                            max_depth: int = None,
                            min_samples_split: int = 2,
                            min_samples_leaf: int = 1,
                            class_weight: str = 'balanced',
                            cv_folds: int = 10) -> Dict[str, Any]:
        """
        Perform Random Forest feature importance analysis.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features matrix
        y_train : pd.Series
            Training target variable
        n_estimators : int
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of the trees
        min_samples_split : int
            Minimum samples required to split an internal node
        min_samples_leaf : int
            Minimum samples required to be at a leaf node
        class_weight : str
            Class weight strategy ('balanced' or None)
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Dictionary containing importance results and model performance
        """
        
        print(f"🌲 Starting Random Forest feature importance analysis...")
        print(f"  📊 Features: {X_train.shape[1]}")
        print(f"  👥 Samples: {X_train.shape[0]}")
        print(f"  🌳 Trees: {n_estimators}")
        
        # Validate inputs
        self._validate_inputs(X_train, y_train)
        
        # Initialize Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            oob_score=True
        )
        
        # Fit the model
        print(f"  🔄 Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        
        # Get feature importances
        importances = self.rf_model.feature_importances_
        feature_names = X_train.columns.tolist()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Calculate additional metrics
        print(f"  📈 Evaluating model performance...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.rf_model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='roc_auc', n_jobs=-1
        )
        
        # OOB score
        oob_score = self.rf_model.oob_score_
        
        # Training predictions for AUC
        y_pred_proba = self.rf_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_pred_proba)
        
        # Package results
        results = {
            'feature_importance': importance_df,
            'model_performance': {
                'train_auc': train_auc,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'oob_score': oob_score
            },
            'model_parameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'class_weight': class_weight,
                'random_state': self.random_state
            },
            'data_info': {
                'n_features': X_train.shape[1],
                'n_samples': X_train.shape[0],
                'feature_names': feature_names,
                'target_distribution': y_train.value_counts().to_dict()
            }
        }
        
        # Store results
        self.importance_results = results
        
        # Print summary
        self._print_importance_summary(results)
        
        return results
    
    def get_top_features(self, 
                        n_features: int = None,
                        importance_threshold: float = None,
                        method: str = 'top_n') -> List[str]:
        """
        Get top features based on importance analysis.
        
        Parameters:
        -----------
        n_features : int, optional
            Number of top features to return
        importance_threshold : float, optional
            Minimum importance threshold
        method : str
            Selection method: 'top_n' or 'threshold'
            
        Returns:
        --------
        List[str]
            List of selected feature names
        """
        
        if self.importance_results is None:
            raise ValueError("No importance analysis results available. Run analyze_rf_importance() first.")
        
        importance_df = self.importance_results['feature_importance']
        
        if method == 'top_n':
            if n_features is None:
                raise ValueError("n_features must be specified for 'top_n' method")
            selected_features = importance_df.head(n_features)['feature'].tolist()
            print(f"🎯 Selected top {len(selected_features)} features by importance")
            
        elif method == 'threshold':
            if importance_threshold is None:
                raise ValueError("importance_threshold must be specified for 'threshold' method")
            selected_features = importance_df[
                importance_df['importance'] >= importance_threshold
            ]['feature'].tolist()
            print(f"🎯 Selected {len(selected_features)} features with importance >= {importance_threshold}")
            
        else:
            raise ValueError(f"Invalid method: {method}")
        
        return selected_features
    
    def plot_importance(self, 
                       top_n: int = 20,
                       save_path: str = None) -> None:
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot
        save_path : str, optional
            Path to save the plot
        """
        
        if self.importance_results is None:
            raise ValueError("No importance analysis results available.")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            importance_df = self.importance_results['feature_importance']
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(10, max(6, top_n * 0.3)))
            sns.barplot(data=top_features, x='importance', y='feature', orient='h')
            plt.title(f'Top {top_n} Feature Importances (Random Forest)')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Importance plot saved to: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("⚠️  Matplotlib/Seaborn not available for plotting")
            # Print text-based plot instead
            self._print_text_importance_plot(top_n)
    
    def compare_importance_methods(self,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        Compare feature importance using multiple methods.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features matrix
        y_train : pd.Series
            Training target variable
            
        Returns:
        --------
        dict
            Dictionary with importance results from different methods
        """
        
        print(f"🔍 Comparing multiple feature importance methods...")
        
        results = {}
        
        # 1. Random Forest (already computed)
        if self.importance_results is not None:
            rf_importance = self.importance_results['feature_importance'].copy()
            rf_importance = rf_importance.rename(columns={'importance': 'rf_importance'})
            results['random_forest'] = rf_importance
        
        # 2. Univariate statistical tests
        try:
            from sklearn.feature_selection import SelectKBest, f_classif
            
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X_train, y_train)
            
            univariate_importance = pd.DataFrame({
                'feature': X_train.columns,
                'f_score': selector.scores_,
                'p_value': selector.pvalues_
            }).sort_values('f_score', ascending=False).reset_index(drop=True)
            
            results['univariate'] = univariate_importance
            
        except Exception as e:
            warnings.warn(f"Failed to compute univariate importance: {str(e)}")
        
        # 3. Mutual Information
        try:
            from sklearn.feature_selection import mutual_info_classif
            
            mi_scores = mutual_info_classif(X_train, y_train, random_state=self.random_state)
            
            mi_importance = pd.DataFrame({
                'feature': X_train.columns,
                'mutual_info': mi_scores
            }).sort_values('mutual_info', ascending=False).reset_index(drop=True)
            
            results['mutual_information'] = mi_importance
            
        except Exception as e:
            warnings.warn(f"Failed to compute mutual information: {str(e)}")
        
        print(f"  ✅ Computed importance using {len(results)} methods")
        
        return results
    
    def _validate_inputs(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Validate input data."""
        if X_train.shape[0] != len(y_train):
            raise ValueError("Feature matrix and target variable must have same number of samples")
        
        if len(y_train.unique()) < 2:
            raise ValueError("Target variable must have at least 2 classes")
        
        # Check for non-numeric features
        non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            warnings.warn(f"Found non-numeric features: {list(non_numeric_cols)}")
    
    def _print_importance_summary(self, results: Dict[str, Any]):
        """Print summary of importance analysis."""
        
        performance = results['model_performance']
        importance_df = results['feature_importance']
        
        print(f"\n🌲 RANDOM FOREST IMPORTANCE SUMMARY")
        print(f"  🎯 Training AUC: {performance['train_auc']:.4f}")
        print(f"  📊 CV AUC: {performance['cv_auc_mean']:.4f} ± {performance['cv_auc_std']:.4f}")
        print(f"  🎲 OOB Score: {performance['oob_score']:.4f}")
        
        print(f"\n  🏆 Top 10 most important features:")
        top_features = importance_df.head(10)
        for i, row in top_features.iterrows():
            print(f"    {i+1:2d}. {row['feature']:20s} {row['importance']:.4f}")
        
        # Importance distribution stats
        importances = importance_df['importance']
        print(f"\n  📈 Importance distribution:")
        print(f"    Mean: {importances.mean():.4f}")
        print(f"    Std:  {importances.std():.4f}")
        print(f"    Max:  {importances.max():.4f}")
        print(f"    Min:  {importances.min():.4f}")
    
    def _print_text_importance_plot(self, top_n: int):
        """Print text-based importance plot."""
        
        if self.importance_results is None:
            return
        
        importance_df = self.importance_results['feature_importance']
        top_features = importance_df.head(top_n)
        
        print(f"\n📊 Top {top_n} Feature Importances:")
        print("  " + "="*60)
        
        max_importance = top_features['importance'].max()
        
        for i, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Create bar visualization
            bar_length = int((importance / max_importance) * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            print(f"  {i+1:2d}. {feature:15s} |{bar}| {importance:.4f}")


def save_importance_results(importance_results: Dict[str, Any], output_path: str):
    """
    Save feature importance results to file.
    
    Parameters:
    -----------
    importance_results : dict
        Feature importance results to save
    output_path : str
        Output file path
    """
    import pickle
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(importance_results, f)
    
    print(f"💾 Feature importance results saved to: {output_path}")
    
    # Also save feature importance as CSV
    if 'feature_importance' in importance_results:
        csv_path = output_path.with_suffix('.csv')
        importance_results['feature_importance'].to_csv(csv_path, index=False)
        print(f"📊 Feature importance table saved to: {csv_path}")


def load_importance_results(results_path: str) -> Dict[str, Any]:
    """
    Load feature importance results from file.
    
    Parameters:
    -----------
    results_path : str
        Path to saved importance results
        
    Returns:
    --------
    dict
        Loaded importance results
    """
    import pickle
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results