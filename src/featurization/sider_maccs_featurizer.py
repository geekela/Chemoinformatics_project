import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys
from typing import Optional


def featurizer_maccs(
    df: pd.DataFrame,
    mol_col: str = 'Molecule',
    include_lipinski: bool = True
) -> pd.DataFrame:
    """
    Calculates MACCS Keys Fingerprints (167 bits) and Lipinski Rule of Five descriptors
    for drug-likeness assessment, particularly suited for SIDER side-effect prediction.

    This featurizer uses MACCS keys instead of Morgan fingerprints. MACCS keys are:
    - Structure-based: 166 predefined structural keys + 1 placeholder
    - Interpretable: Each bit represents a specific molecular fragment
    - Compact: Only 167 bits vs 2048 for Morgan
    - Drug-focused: Designed for pharmaceutical compound comparison

    Args:
        df (pd.DataFrame): DataFrame containing RDKit Mol objects.
        mol_col (str): Name of the column containing the Mol objects. Default: 'Molecule'.
        include_lipinski (bool): Whether to include Lipinski descriptors. Default: True.

    Returns:
        pd.DataFrame: DataFrame with original columns plus:
            - 167 MACCS fingerprint columns (MACCS_0 to MACCS_166)
            - 10 Lipinski/drug-likeness descriptors (if include_lipinski=True)

    Notes:
        - MACCS keys are particularly useful for similarity searches in drug discovery
        - Lipinski descriptors help assess "drug-likeness" and bioavailability
        - The function handles None/invalid molecules by filling with zeros/NaN
    """

    df = df.copy()  # Work on a copy to avoid modifying the original

    # --- 1. LIPINSKI DESCRIPTORS (Drug-Likeness Features) ---
    if include_lipinski:
        lipinski_descriptors = {
            'MolecularWeight': lambda mol: Descriptors.MolWt(mol) if mol else np.nan,
            'LogP': lambda mol: Descriptors.MolLogP(mol) if mol else np.nan,
            'NumHDonors': lambda mol: Lipinski.NumHDonors(mol) if mol else np.nan,
            'NumHAcceptors': lambda mol: Lipinski.NumHAcceptors(mol) if mol else np.nan,
            'NumRotatableBonds': lambda mol: Descriptors.NumRotatableBonds(mol) if mol else np.nan,
            'NumAromaticRings': lambda mol: Descriptors.NumAromaticRings(mol) if mol else np.nan,
            'TPSA': lambda mol: Descriptors.TPSA(mol) if mol else np.nan,
            'NumHeteroatoms': lambda mol: Descriptors.NumHeteroatoms(mol) if mol else np.nan,
            'NumSaturatedRings': lambda mol: Descriptors.NumSaturatedRings(mol) if mol else np.nan,
            'FractionCSP3': lambda mol: Descriptors.FractionCSP3(mol) if mol else np.nan,
        }

        print(f"Calculating {len(lipinski_descriptors)} Lipinski/drug-likeness descriptors...")

        for desc_name, desc_func in lipinski_descriptors.items():
            df[desc_name] = df[mol_col].apply(desc_func)

    # --- 2. MACCS KEYS FINGERPRINTS ---
    print("Generating MACCS Keys fingerprints (167 bits)...")

    def get_maccs_fingerprint(mol):
        """
        Generate MACCS keys fingerprint for a molecule.

        Args:
            mol: RDKit Mol object

        Returns:
            list: List of 167 binary values (0 or 1)
        """
        if mol is None:
            return [0] * 167

        try:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            # Convert to list (note: MACCS has 167 keys)
            return list(maccs_fp)
        except Exception as e:
            print(f"Warning: Could not generate MACCS fingerprint: {e}")
            return [0] * 167

    # Generate fingerprints for all molecules
    maccs_arrays = df[mol_col].apply(get_maccs_fingerprint).tolist()

    # Create column names for MACCS fingerprint bits
    maccs_col_names = [f'MACCS_{i}' for i in range(167)]

    # Create DataFrame from fingerprint arrays
    df_maccs = pd.DataFrame(maccs_arrays, columns=maccs_col_names, index=df.index)

    # Concatenate MACCS features to main DataFrame
    df = pd.concat([df, df_maccs], axis=1)

    # --- 3. REORGANIZE COLUMNS ---
    # Move Molecule column to position 1 (after the first column, usually SMILES/ID)
    if mol_col in df.columns:
        col_to_move = df.pop(mol_col)
        df.insert(1, mol_col, col_to_move)

    # --- 4. SUMMARY REPORT ---
    num_lipinski = len(lipinski_descriptors) if include_lipinski else 0
    num_maccs = 167
    total_features = num_lipinski + num_maccs

    print(f"\nMACCS Featurization Complete!")
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Features added:")
    print(f"  - Lipinski descriptors: {num_lipinski}")
    print(f"  - MACCS Keys bits: {num_maccs}")
    print(f"  - Total features: {total_features}\n")

    return df
