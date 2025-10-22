import deepchem as dc
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def featurize_molecules(
    df,
    smiles_col="canonical_smiles",
    target_col="pIC50",
    methods=None,
    remove_nan=True,
    remove_zero_variance=True,
    concatenate=False,
    return_dataframe=True):

    # Hide RDKit warning    
    rdkit_logger = logging.getLogger('rdkit')
    original_level = rdkit_logger.level
    rdkit_logger.setLevel(logging.ERROR)

    if methods is None:
        methods = ["rdkit", "maccs"]

    smiles=df[ smiles_col].tolist()
    target=df[target_col].tolist()

    all_features = {}
    for method in methods:
        if method == "rdkit":
            featurizer = dc.feat.RDKitDescriptors()
        elif method == "maccs":
            featurizer = dc.feat.MACCSKeysFingerprint()
        elif method == "ecfp":
            featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)
        elif method == "graphconv":
            featurizer = dc.feat.ConvMolFeaturizer()
        else:
            raise ValueError(f"Unknown featurizer: {method}")

        X = featurizer.featurize(smiles)
        print(f"{method.upper()} features shape before cleaning: {np.shape(X)}")

        if isinstance(X, np.ndarray):
            if remove_nan:
                X = X[:, ~np.isnan(X).any(axis=0)]
            if remove_zero_variance:
                X = VarianceThreshold(threshold=0.0).fit_transform(X)

        print(f"{method.upper()} features shape after cleaning: {np.shape(X)}")
        if isinstance(X, np.ndarray):
            feature_names = [f"{method}_{i}" for i in range(X.shape[1])]
            df_feat = pd.DataFrame(X, columns=feature_names)
        else:
            df_feat = pd.DataFrame({f"{method}_features": X})

        # Add SMILES and target
        df_feat.insert(0, smiles_col, df[smiles_col].values)
        if target is not None:
            df_feat.insert(1, target_col, target)

        all_features[method] = df_feat

    # Restore warning    
    rdkit_logger.setLevel(original_level)

        
    # Concatenate if requested
    if concatenate:
        df_concat = df[[smiles_col]].copy()
        if target is not None:
            df_concat[target_col] = target

        for feat_df in all_features.values():
            df_concat = pd.concat(
                [df_concat, feat_df.drop(columns=[smiles_col, target_col])],
                axis=1
            )

        print(f"Final concatenated feature matrix shape: {df_concat.shape}")
        return df_concat

    else:
        return all_features
