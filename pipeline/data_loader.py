"""
Data Loader Module
==================

Handles data loading, preprocessing, and matrix preparation
for protein-based machine learning analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import warnings


class DataLoader:
    """
    Data loader for protein expression and phenotype data.

    Reproduces R get_matrices functionality with Python.
    """

    def __init__(self, base_path: str):
        """
        Initialize data loader.

        Parameters
        ----------
        base_path : str
            Base path for data files
        """
        self.base_path = Path(base_path)
        self.data_cache = {}

    # ---------- internal helpers ----------

    def _read_with_sample_index(self, path: str) -> pd.DataFrame:
        """
        Read a CSV and set 'sample_id' as the index if present.
        Falls back to common legacy first-column index patterns.
        """
        df = pd.read_csv(path)
        if "sample_id" in df.columns:
            return df.set_index("sample_id")
        # Legacy/dirty CSVs: if first column looks like an index, use it
        first_col = df.columns[0]
        if first_col.lower() in ("unnamed: 0", "index", "sample_id"):
            return df.set_index(first_col).rename_axis("sample_id")
        warnings.warn(
            f"'sample_id' column not found in {path}. "
            "Keeping a default RangeIndex; downstream alignment may drop samples."
        )
        return df

    def _find_covariates(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Return a mapping {canonical_name -> actual_column_name} for covariates
        present in df. Supports common aliases/casing.
        """
        aliases = {
            "sex": ["sex", "Sex"],
            "age": ["age", "Age", "Age_at_recruitment", "age_at_recruitment"],
            "bmi": ["bmi", "BMI"],
        }
        present = {}
        lower_map = {c.lower(): c for c in df.columns}
        for canon, candidates in aliases.items():
            for cand in candidates:
                # exact match
                if cand in df.columns:
                    present[canon] = cand
                    break
                # case-insensitive fallback
                if cand.lower() in lower_map:
                    present[canon] = lower_map[cand.lower()]
                    break
        return present

    def _align_on_index(
        self, features: pd.DataFrame, phenotypes: pd.DataFrame, dataset_name: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align features and phenotypes to the intersection of their indices,
        preserving identical ordering. Warns if samples are dropped.
        """
        # If already perfectly aligned, keep as is
        if features.index.equals(phenotypes.index):
            return features, phenotypes

        common = features.index.intersection(phenotypes.index)
        if len(common) == 0:
            raise ValueError(
                f"No overlapping sample_ids between features and phenotypes for {dataset_name}."
            )

        if (len(common) < len(features)) or (len(common) < len(phenotypes)):
            warnings.warn(
                f"{dataset_name}: dropping non-overlapping samples "
                f"(features {len(features)} → {len(common)}, "
                f"phenotypes {len(phenotypes)} → {len(common)})."
            )

        # Reindex both to the same order (intersection sorted by appearance in features)
        common_ordered = features.index[features.index.isin(common)]
        return features.loc[common_ordered], phenotypes.loc[common_ordered]

    def _handle_missing_values(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Handle missing values in data."""
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"  ⚠️  Found {missing_count} missing values in {name}")
            # Simple imputation (can be improved later)
            if df.select_dtypes(include=[np.number]).shape[1] > 0:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            if df.select_dtypes(include=["object", "category"]).shape[1] > 0:
                cat_cols = df.select_dtypes(include=["object", "category"]).columns
                for col in cat_cols:
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "missing")
        return df

    def _print_data_summary(self, data: Dict[str, Any]):
        """Print summary of loaded data."""
        info = data["data_info"]

        print(f"\n📊 DATA LOADING SUMMARY")
        print(f"  🎯 Prediction mode: {info['pred_mode']}")
        print(
            f"  🏋️  Training: {info['train_dimensions'][0]} samples × {info['train_dimensions'][1]} features"
        )
        print(
            f"  🧪 Validation: {info['val_dimensions'][0]} samples × {info['val_dimensions'][1]} features"
        )
        print(f"  📋 Target variable: {info['target_column']}")
        print(f"  🎪 Training target distribution: {info['target_distribution_train']}")
        print(f"  🎪 Validation target distribution: {info['target_distribution_val']}")

        if len(info["feature_names"]) <= 10:
            print(f"  🧬 Features: {info['feature_names']}")
        else:
            print(f"  🧬 Features: {info['feature_names'][:5]} ... {info['feature_names'][-5:]}")

    # ---------- public API ----------

    def load_matrices(
        self,
        protein_train_path: str,
        phenotype_train_path: str,
        protein_val_path: str,
        phenotype_val_path: str,
        pred_mode: str = "prot_only",
        target_column: str = "oa_status",
    ) -> Dict[str, Any]:
        """
        Load and prepare protein and phenotype matrices.

        Parameters
        ----------
        protein_train_path : str
            Path to training protein expression data
        phenotype_train_path : str
            Path to training phenotype data
        protein_val_path : str
            Path to validation protein expression data
        phenotype_val_path : str
            Path to validation phenotype data
        pred_mode : str
            Prediction mode: 'prot_only', 'sexagebmi', or 'comb'
        target_column : str
            Name of target variable column

        Returns
        -------
        dict
            Dictionary containing loaded matrices and metadata
        """

        print(f"📥 Loading data for prediction mode: {pred_mode}")

        # Load protein expression data
        print("  📊 Loading protein expression matrices...")
        p_mtx_train = self._read_with_sample_index(protein_train_path)
        p_mtx_val = self._read_with_sample_index(protein_val_path)

        # Load phenotype data
        print("  📋 Loading phenotype data...")
        pheno_train = self._read_with_sample_index(phenotype_train_path)
        pheno_val = self._read_with_sample_index(phenotype_val_path)

        # Align on sample_id index (drops non-overlapping rows with warning)
        p_mtx_train, pheno_train = self._align_on_index(p_mtx_train, pheno_train, "training")
        p_mtx_val, pheno_val = self._align_on_index(p_mtx_val, pheno_val, "validation")

        # Check target variable
        if target_column not in pheno_train.columns:
            raise ValueError(f"Target column '{target_column}' not found in training phenotype data")
        if target_column not in pheno_val.columns:
            warnings.warn(
                f"Target column '{target_column}' not found in validation phenotypes. "
                "This is fine if you're doing pure prediction on val without labels."
            )

        # Prepare matrices based on prediction mode
        if pred_mode == "prot_only":
            # Only protein features
            # Exclude any covariate-looking columns if present in protein matrix
            covs_in_p = self._find_covariates(p_mtx_train)
            feature_cols = [c for c in p_mtx_train.columns if c not in set(covs_in_p.values())]
            p_mtx_train_final = p_mtx_train[feature_cols]
            p_mtx_val_final = p_mtx_val[feature_cols]

        elif pred_mode == "sexagebmi":
            # Only demographic covariates (prefer taking them from the protein matrix if already merged)
            covs_in_p = self._find_covariates(p_mtx_train)
            covs_in_ph = self._find_covariates(pheno_train)

            if covs_in_p:
                cols = list(covs_in_p.values())
                p_mtx_train_final = p_mtx_train[cols]
                p_mtx_val_final = p_mtx_val[cols]
            elif covs_in_ph:
                cols = list(covs_in_ph.values())
                p_mtx_train_final = pheno_train[cols]
                p_mtx_val_final = pheno_val[cols]
            else:
                raise ValueError("No demographic covariates (sex/age/bmi) found in data.")

        elif pred_mode == "comb":
            # Combined: proteins + demographics
            covs_in_p = self._find_covariates(p_mtx_train)
            covs_in_ph = self._find_covariates(pheno_train)

            # protein columns should exclude covariates if they live in the protein table
            protein_cols = (
                [c for c in p_mtx_train.columns if c not in set(covs_in_p.values())]
                if covs_in_p
                else list(p_mtx_train.columns)
            )

            if covs_in_p:
                all_features = protein_cols + list(covs_in_p.values())
                p_mtx_train_final = p_mtx_train[all_features]
                p_mtx_val_final = p_mtx_val[all_features]
            elif covs_in_ph:
                cov_cols = list(covs_in_ph.values())
                p_mtx_train_final = pd.concat([p_mtx_train[protein_cols], pheno_train[cov_cols]], axis=1)
                p_mtx_val_final = pd.concat([p_mtx_val[protein_cols], pheno_val[cov_cols]], axis=1)
            else:
                warnings.warn("No demographic covariates found; falling back to protein-only.")
                p_mtx_train_final = p_mtx_train[protein_cols]
                p_mtx_val_final = p_mtx_val[protein_cols]
        else:
            raise ValueError(f"Invalid pred_mode: {pred_mode}")

        # Handle missing values
        p_mtx_train_final = self._handle_missing_values(p_mtx_train_final, "training features")
        p_mtx_val_final = self._handle_missing_values(p_mtx_val_final, "validation features")
        pheno_train = self._handle_missing_values(pheno_train, "training phenotypes")
        pheno_val = self._handle_missing_values(pheno_val, "validation phenotypes")

        # Create result dictionary
        result = {
            "p_mtx_traintest": p_mtx_train_final,
            "pheno_train_test": pheno_train,
            "p_mtx_val": p_mtx_val_final,
            "pheno_val": pheno_val,
            "data_info": {
                "pred_mode": pred_mode,
                "target_column": target_column,
                "train_dimensions": p_mtx_train_final.shape,
                "val_dimensions": p_mtx_val_final.shape,
                "n_features": p_mtx_train_final.shape[1],
                "train_samples": p_mtx_train_final.shape[0],
                "val_samples": p_mtx_val_final.shape[0],
                "feature_names": list(p_mtx_train_final.columns),
                "target_distribution_train": pheno_train[target_column].value_counts().to_dict()
                if target_column in pheno_train.columns
                else {},
                "target_distribution_val": pheno_val[target_column].value_counts().to_dict()
                if target_column in pheno_val.columns
                else {},
            },
        }

        # Print summary
        self._print_data_summary(result)

        return result


def create_sample_data(
    output_dir: str,
    n_samples_train: int = 2963,
    n_samples_val: int = 328,
    n_proteins: int = 2131,
    random_state: int = 42,
) -> Dict[str, str]:
    """
    Create sample data files for testing the pipeline.

    Parameters
    ----------
    output_dir : str
        Directory to save sample data
    n_samples_train : int
        Number of training samples
    n_samples_val : int
        Number of validation samples
    n_proteins : int
        Number of proteins
    random_state : int
        Random seed

    Returns
    -------
    dict
        Dictionary with paths to created files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(random_state)

    # Generate sample IDs
    train_ids = [f"sample_{i:06d}" for i in range(n_samples_train)]
    val_ids = [f"sample_{i:06d}" for i in range(n_samples_train, n_samples_train + n_samples_val)]

    # Generate protein names
    protein_names = [f"P{i+1}" for i in range(n_proteins)]

    # Generate protein expression data (log-normal distribution)
    protein_train = pd.DataFrame(
        np.random.lognormal(0, 1, (n_samples_train, n_proteins)),
        columns=protein_names,
    )
    protein_train.insert(0, "sample_id", train_ids)  # explicit column for IDs

    protein_val = pd.DataFrame(
        np.random.lognormal(0, 1, (n_samples_val, n_proteins)),
        columns=protein_names,
    )
    protein_val.insert(0, "sample_id", val_ids)

    # Generate phenotype data
    def create_phenotype_data(sample_ids, n_samples):
        data = {
            "sample_id": sample_ids,
            "Age_at_recruitment": np.random.normal(55, 15, n_samples).astype(int).clip(20, 85),
            "Sex": np.random.choice(["Male", "Female"], n_samples),
            "mean_NPX": np.random.normal(0, 1, n_samples),
            "Plate0": np.random.choice([0, 1], n_samples),
            "Plate2": np.random.normal(0, 1, n_samples),
            "Plate3": np.random.normal(0, 1, n_samples),
            "bmi": np.random.normal(25, 5, n_samples).clip(15, 50),
            "season": np.random.choice(["Spring", "Summer", "Fall", "Winter"], n_samples),
            "oa_status": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        }
        data["Age2"] = data["Age_at_recruitment"] ** 2
        return pd.DataFrame(data)

    pheno_train = create_phenotype_data(train_ids, n_samples_train)
    pheno_val = create_phenotype_data(val_ids, n_samples_val)

    # Save all without index
    files = {
        "protein_train": output_path / "protein_train_sample.csv",
        "protein_val": output_path / "protein_val_sample.csv",
        "phenotype_train": output_path / "phenotype_train_sample.csv",
        "phenotype_val": output_path / "phenotype_val_sample.csv",
    }

    protein_train.to_csv(files["protein_train"], index=False)
    protein_val.to_csv(files["protein_val"], index=False)
    pheno_train.to_csv(files["phenotype_train"], index=False)
    pheno_val.to_csv(files["phenotype_val"], index=False)

    print(f"📁 Sample data created in: {output_dir}")
    print(f"📊 Training: {n_samples_train} samples × {n_proteins} proteins")
    print(f"📊 Validation: {n_samples_val} samples × {n_proteins} proteins")

    return {k: str(v) for k, v in files.items()}
