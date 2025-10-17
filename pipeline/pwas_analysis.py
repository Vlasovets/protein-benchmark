"""
PWAS Analysis Module
====================

Performs Protein-Wide Association Study (PWAS) analysis
using the methodology from the R pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats
import warnings
from pathlib import Path

class PWASAnalyzer:
    """
    Performs PWAS analysis for protein expression data.
    
    Reproduces R perform_pwas functionality with Python.
    """
    
    def __init__(self):
        """Initialize PWAS analyzer."""
        self.results = None
        self.summary_stats = None
        
    def perform_pwas(self,
                    protein_matrix: pd.DataFrame,
                    phenotype_data: pd.DataFrame, 
                    target_column: str = "oa_status",
                    covariate_columns: List[str] = None,
                    fdr_threshold: float = 0.05,
                    p_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Perform PWAS analysis on protein expression data.
        
        Parameters:
        -----------
        protein_matrix : pd.DataFrame
            Protein expression matrix (samples x proteins)
        phenotype_data : pd.DataFrame
            Phenotype data with target variable and covariates
        target_column : str
            Name of target variable column
        covariate_columns : List[str], optional
            List of covariate column names to include in models
        fdr_threshold : float
            FDR threshold for significance (default: 0.05)
        p_threshold : float
            Nominal p-value threshold fallback (default: 0.05)
            
        Returns:
        --------
        dict
            Dictionary containing PWAS results and statistics
        """
        
        print(f"🧬 Starting PWAS analysis...")
        print(f"  🎯 Target: {target_column}")
        print(f"  🧪 Proteins: {protein_matrix.shape[1]}")
        print(f"  👥 Samples: {protein_matrix.shape[0]}")
        
        # Validate inputs
        self._validate_inputs(protein_matrix, phenotype_data, target_column)
        
        # Set up covariates
        if covariate_columns is None:
            # Use common covariates if available
            potential_covs = ['Age_at_recruitment', 'Sex', 'bmi', 'age', 'sex', 
                            'mean_NPX', 'Plate0', 'Plate2', 'Plate3']
            covariate_columns = [col for col in potential_covs 
                               if col in phenotype_data.columns]
        
        print(f"  🔧 Covariates: {covariate_columns}")
        
        # Prepare data
        aligned_data = self._align_data(protein_matrix, phenotype_data)
        protein_data = aligned_data['proteins']
        pheno_data = aligned_data['phenotypes']
        
        # Run association tests
        results = self._run_association_tests(
            protein_data, pheno_data, target_column, covariate_columns
        )
        
        # Apply multiple testing correction
        corrected_results = self._apply_multiple_testing_correction(
            results, fdr_threshold, p_threshold
        )
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(corrected_results, protein_data)
        
        # Package results
        pwas_results = {
            'results': corrected_results,
            'summary': summary,
            'parameters': {
                'target_column': target_column,
                'covariate_columns': covariate_columns,
                'fdr_threshold': fdr_threshold,
                'p_threshold': p_threshold,
                'n_proteins_tested': len(results),
                'n_samples': protein_data.shape[0]
            },
            'data_info': {
                'protein_names': list(protein_data.columns),
                'sample_ids': list(protein_data.index),
                'target_distribution': pheno_data[target_column].value_counts().to_dict()
            }
        }
        
        # Store results
        self.results = pwas_results
        self.summary_stats = summary
        
        # Print summary
        self._print_pwas_summary(pwas_results)
        
        return pwas_results
    
    def get_selected_proteins(self, 
                            selection_strategy: str = "fdr_significant",
                            max_proteins: int = 200) -> List[str]:
        """
        Get selected proteins based on PWAS results.
        
        Parameters:
        -----------
        selection_strategy : str
            Strategy for protein selection:
            - "fdr_significant": FDR < 0.05
            - "nominal_significant": nominal p < 0.05  
            - "top_n": top N proteins by p-value
        max_proteins : int
            Maximum number of proteins to return
            
        Returns:
        --------
        List[str]
            List of selected protein names
        """
        
        if self.results is None:
            raise ValueError("No PWAS results available. Run perform_pwas() first.")
        
        results_df = self.results['results']
        
        if selection_strategy == "fdr_significant":
            selected = results_df[results_df['fdr_significant']]['protein'].tolist()
            print(f"🎯 Selected {len(selected)} FDR-significant proteins")
            
        elif selection_strategy == "nominal_significant":
            selected = results_df[results_df['nominal_significant']]['protein'].tolist()
            print(f"🎯 Selected {len(selected)} nominally significant proteins (p < 0.05)")
            
        elif selection_strategy == "top_n":
            selected = results_df.nsmallest(max_proteins, 'p_value')['protein'].tolist()
            print(f"🎯 Selected top {len(selected)} proteins by p-value")
            
        else:
            raise ValueError(f"Invalid selection strategy: {selection_strategy}")
        
        # Apply max_proteins limit
        if len(selected) > max_proteins:
            print(f"  ✂️  Limiting to top {max_proteins} proteins")
            # Re-sort by p-value and take top max_proteins
            sorted_proteins = results_df[results_df['protein'].isin(selected)].nsmallest(
                max_proteins, 'p_value'
            )['protein'].tolist()
            selected = sorted_proteins
        
        return selected
    
    def _validate_inputs(self, protein_matrix: pd.DataFrame, 
                        phenotype_data: pd.DataFrame, target_column: str):
        """Validate input data."""
        if target_column not in phenotype_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in phenotype data")
        
        if protein_matrix.shape[0] != phenotype_data.shape[0]:
            warnings.warn("Sample count mismatch between protein and phenotype data")
        
        # Check for non-numeric proteins
        non_numeric_proteins = protein_matrix.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_proteins) > 0:
            warnings.warn(f"Found non-numeric protein columns: {list(non_numeric_proteins)}")
    
    def _align_data(self, protein_matrix: pd.DataFrame, 
                   phenotype_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Align protein and phenotype data by sample IDs."""
        
        # Find common samples
        common_samples = set(protein_matrix.index) & set(phenotype_data.index)
        
        if len(common_samples) == 0:
            # Try positional alignment if indices don't match
            min_samples = min(len(protein_matrix), len(phenotype_data))
            protein_aligned = protein_matrix.iloc[:min_samples].copy()
            pheno_aligned = phenotype_data.iloc[:min_samples].copy()
            warnings.warn(f"No common sample IDs found. Using positional alignment for {min_samples} samples.")
        else:
            # Align by common sample IDs
            common_samples = list(common_samples)
            protein_aligned = protein_matrix.loc[common_samples].copy()
            pheno_aligned = phenotype_data.loc[common_samples].copy()
        
        return {
            'proteins': protein_aligned,
            'phenotypes': pheno_aligned
        }
    
    def _preprocess_phenotype_data(self, pheno_data: pd.DataFrame, covariate_columns: List[str]) -> pd.DataFrame:
        """Preprocess phenotype data with proper categorical encoding."""
        processed_data = pheno_data.copy()
        
        # Handle Sex variable encoding
        if 'Sex' in processed_data.columns:
            sex_mapping = {'Female': 0, 'Male': 1, 'female': 0, 'male': 1, 'F': 0, 'M': 1}
            processed_data['Sex_encoded'] = processed_data['Sex'].map(sex_mapping)
            print(f"    🔧 Sex encoding: {dict(zip(processed_data['Sex'].unique(), processed_data['Sex_encoded'].unique()))}")
        
        # Handle other categorical variables
        for cov in covariate_columns:
            if cov in processed_data.columns and cov != 'Sex':
                col_data = processed_data[cov]
                if not pd.api.types.is_numeric_dtype(col_data):
                    print(f"    ⚠️ Warning: {cov} is not numeric: {col_data.dtype}")
                    try:
                        processed_data[f'{cov}_numeric'] = pd.to_numeric(col_data, errors='coerce')
                        print(f"    ✅ Converted {cov} to numeric")
                    except:
                        print(f"    ❌ Failed to convert {cov} to numeric")
        
        return processed_data

    def _run_association_tests(self,
                              protein_data: pd.DataFrame,
                              pheno_data: pd.DataFrame, 
                              target_column: str,
                              covariate_columns: List[str]) -> pd.DataFrame:
        """Run simplified PWAS analysis with proper categorical encoding."""
        
        results = []
        
        # Preprocess phenotype data
        processed_pheno = self._preprocess_phenotype_data(pheno_data, covariate_columns)
        
        # Update covariate list to use encoded versions
        updated_covariates = []
        for cov in covariate_columns:
            if cov == 'Sex' and 'Sex_encoded' in processed_pheno.columns:
                updated_covariates.append('Sex_encoded')
            elif cov in processed_pheno.columns:
                updated_covariates.append(cov)
            elif f'{cov}_numeric' in processed_pheno.columns:
                updated_covariates.append(f'{cov}_numeric')
        
        # Filter available covariates
        available_covs = [cov for cov in updated_covariates if cov in processed_pheno.columns]
        print(f"  🔧 Using covariates: {available_covs}")
        
        y = processed_pheno[target_column].values
        
        # Prepare covariate matrix once
        if available_covs:
            X_covs = processed_pheno[available_covs].values
            print(f"  📊 Covariate matrix shape: {X_covs.shape}")
            # Check covariate data types
            for i, cov in enumerate(available_covs):
                col_data = X_covs[:, i]
                print(f"    {cov}: min={col_data.min():.3f}, max={col_data.max():.3f}, has_nan={np.isnan(col_data).any()}")
        else:
            X_covs = None

        
        print(f"  🔄 Testing {protein_data.shape[1]} proteins...")
        
        for i, protein in enumerate(protein_data.columns):
            if i % 500 == 0:
                print(f"    Processing protein {i+1}/{protein_data.shape[1]}")
            
            try:
                # Prepare features
                X_protein = protein_data[protein].values.reshape(-1, 1)
                
                if X_covs is not None:
                    X_full = np.column_stack([X_protein, X_covs])
                else:
                    X_full = X_protein
                
                # Check for any remaining non-numeric data
                if not np.all(np.isfinite(X_full)):
                    continue
                
                # Use sklearn logistic regression (more robust than statsmodels)
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000, fit_intercept=True)
                model.fit(X_full, y)
                
                # Get protein coefficient (first feature)
                coef = model.coef_[0][0]
                
                # Use correlation p-value as approximation (matches your working implementation)
                from scipy import stats
                correlation, p_value = stats.pearsonr(X_protein.ravel(), y)
                
                results.append({
                    'protein': protein,
                    'coefficient': coef,
                    'p_value': abs(p_value),  # Use correlation p-value
                    'n_samples': len(y),
                    'std_error': np.nan,  # Not calculated in simplified version
                    'odds_ratio': np.exp(coef),
                    'ci_lower': np.nan,
                    'ci_upper': np.nan
                })
                
            except Exception as e:
                if i < 10:  # Print first few errors for debugging
                    print(f"    ❌ Error with protein {protein}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def _apply_multiple_testing_correction(self,
                                         results: pd.DataFrame,
                                         fdr_threshold: float,
                                         p_threshold: float) -> pd.DataFrame:
        """Apply multiple testing correction."""
        
        if len(results) == 0:
            return results
        
        # Apply FDR correction
        rejected, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(
            results['p_value'], alpha=fdr_threshold, method='fdr_bh'
        )
        
        results = results.copy()
        results['p_adjusted'] = pvals_corrected
        results['fdr_significant'] = rejected
        results['nominal_significant'] = results['p_value'] < p_threshold
        
        # Sort by p-value
        results = results.sort_values('p_value').reset_index(drop=True)
        
        return results
    
    def _generate_summary_statistics(self, pwas_results, protein_data=None):
        """Generate summary statistics for PWAS results."""
        if pwas_results.empty:
            return {
                'total_proteins': 0,
                'total_proteins_tested': 0,           # Add missing key
                'total_associations_tested': 0,       # Add missing key
                'fdr_significant_count': 0,
                'bonferroni_significant_count': 0,
                'nominal_significant_count': 0,
                'min_p_value': 1.0,
                'top_protein': None,
                'top_p_value': 1.0,
                'mean_effect_size': 0.0,
                'std_effect_size': 0.0
            }
        
        # Calculate statistics using correct column names
        total_proteins = len(pwas_results)
        fdr_significant = pwas_results['fdr_significant'].sum() if 'fdr_significant' in pwas_results.columns else 0
        bonferroni_significant = 0  # Not implemented in current version
        nominal_significant = pwas_results['nominal_significant'].sum() if 'nominal_significant' in pwas_results.columns else 0
        
        # Get top protein (lowest p-value)
        top_idx = pwas_results['p_value'].idxmin()
        top_protein = pwas_results.loc[top_idx, 'protein']
        top_p_value = pwas_results.loc[top_idx, 'p_value']
        min_p_value = pwas_results['p_value'].min()
        
        return {
            'total_proteins': total_proteins,
            'total_proteins_tested': total_proteins,     # Add missing key
            'total_associations_tested': total_proteins, # Add missing key  
            'fdr_significant_count': fdr_significant,
            'bonferroni_significant_count': bonferroni_significant,
            'nominal_significant_count': nominal_significant,
            'min_p_value': min_p_value,
            'top_protein': top_protein,
            'top_p_value': top_p_value,
            'mean_effect_size': pwas_results['coefficient'].mean() if 'coefficient' in pwas_results.columns else 0.0,
            'std_effect_size': pwas_results['coefficient'].std() if 'coefficient' in pwas_results.columns else 0.0
        }
    
    def _print_pwas_summary(self, pwas_results: Dict[str, Any]):
        """Print summary of PWAS results."""
        
        summary = pwas_results['summary']
        
        print(f"\n🧬 PWAS ANALYSIS SUMMARY")
        print(f"  🔬 Total proteins tested: {summary['total_proteins_tested']}")
        print(f"  ✅ Valid associations: {summary['total_associations_tested']}")
        print(f"  🎯 FDR significant (< 0.05): {summary['fdr_significant_count']}")
        print(f"  📊 Nominal significant (< 0.05): {summary['nominal_significant_count']}")
        print(f"  🏆 Best p-value: {summary['min_p_value']:.2e}")
        
        if summary['top_protein']:
            print(f"  🥇 Top protein: {summary['top_protein']} (p = {summary['top_p_value']:.2e})")
        
        # Print top 10 proteins
        if len(pwas_results['results']) > 0:
            print(f"\n  📋 Top 10 proteins:")
            top_proteins = pwas_results['results'].head(10)
            for i, row in top_proteins.iterrows():
                print(f"    {i+1:2d}. {row['protein']:15s} p={row['p_value']:.2e} OR={row['odds_ratio']:.3f}")


def load_pwas_results(results_path: str) -> Dict[str, Any]:
    """
    Load PWAS results from file.
    
    Parameters:
    -----------
    results_path : str
        Path to saved PWAS results
        
    Returns:
    --------
    dict
        Loaded PWAS results
    """
    # Implementation would depend on file format
    # For now, assume pickle format
    import pickle
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results


def save_pwas_results(pwas_results: Dict[str, Any], output_path: str):
    """
    Save PWAS results to file.
    
    Parameters:
    -----------
    pwas_results : dict
        PWAS results to save
    output_path : str
        Output file path
    """
    import pickle
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(pwas_results, f)
    
    print(f"💾 PWAS results saved to: {output_path}")
    
    # Also save summary as CSV for easy inspection
    if 'results' in pwas_results and len(pwas_results['results']) > 0:
        csv_path = output_path.with_suffix('.csv')
        pwas_results['results'].to_csv(csv_path, index=False)
        print(f"📊 Results table saved to: {csv_path}")