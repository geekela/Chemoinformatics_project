"""
Simple molecular featurizer using RDKit only (no DeepChem)
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.feature_selection import VarianceThreshold


def generate_ecfp(smiles, radius=2, n_bits=1024):
    """Generate Extended Connectivity Fingerprints (ECFP)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def generate_maccs(smiles):
    """Generate MACCS keys fingerprints"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def generate_rdkit_descriptors(smiles):
    """Generate RDKit 2D descriptors"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(200)  # Placeholder

    # Get all descriptor functions
    descriptor_names = [name for name, func in Descriptors.descList]
    descriptors = []

    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            descriptors.append(val)
        except:
            descriptors.append(0.0)

    return np.array(descriptors)


def featurize_molecules_simple(df, smiles_col='canonical_smiles', target_col='pIC50',
                               methods=None):
    """
    Simple featurization using only RDKit

    Args:
        df: DataFrame with SMILES and target
        smiles_col: Column name for SMILES
        target_col: Column name for target variable
        methods: List of methods ('ecfp', 'maccs', 'rdkit')

    Returns:
        Dictionary of {method_name: DataFrame}
    """
    if methods is None:
        methods = ['ecfp', 'maccs', 'rdkit']

    smiles_list = df[smiles_col].tolist()
    target_list = df[target_col].tolist()

    features_dict = {}

    for method in methods:
        print(f"\n   Generating {method.upper()} features...")

        if method == 'ecfp':
            features = np.array([generate_ecfp(s) for s in smiles_list])
        elif method == 'maccs':
            features = np.array([generate_maccs(s) for s in smiles_list])
        elif method == 'rdkit':
            features = np.array([generate_rdkit_descriptors(s) for s in smiles_list])
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"     Shape before cleaning: {features.shape}")

        # Remove NaN columns
        nan_mask = ~np.isnan(features).any(axis=0)
        features = features[:, nan_mask]

        # Remove zero variance features
        if features.shape[1] > 0:
            selector = VarianceThreshold(threshold=0.0)
            features = selector.fit_transform(features)

        print(f"     Shape after cleaning: {features.shape}")

        # Create DataFrame
        feature_names = [f"{method}_{i}" for i in range(features.shape[1])]
        df_feat = pd.DataFrame(features, columns=feature_names)

        # Add SMILES and target
        df_feat.insert(0, smiles_col, df[smiles_col].values)
        df_feat.insert(1, target_col, target_list)

        features_dict[method] = df_feat

    return features_dict
