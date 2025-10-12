import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem
from typing import Optional


def featurizer_morgan(
    df: pd.DataFrame,
    mol_col: str = 'Molecule',
    radius: int = 2,
    n_bits: int = 2048,
    include_descriptors: bool = True
) -> pd.DataFrame:
    """
    Calculates Morgan Fingerprints (circular fingerprints) and molecular descriptors
    for drug side-effect prediction on the SIDER dataset.

    Morgan fingerprints (ECFP - Extended Connectivity Fingerprints):
    - Topology-based: Captures atom neighborhoods at specified radius
    - Flexible: Configurable radius and bit length
    - Comprehensive: Encodes diverse substructural patterns
    - Similarity-focused: Excellent for structure-activity relationships

    Args:
        df (pd.DataFrame): DataFrame containing RDKit Mol objects.
        mol_col (str): Name of the column containing the Mol objects. Default: 'Molecule'.
        radius (int): Radius for Morgan fingerprint (2 = ECFP4, 3 = ECFP6). Default: 2.
        n_bits (int): Number of bits in the fingerprint. Default: 2048.
        include_descriptors (bool): Whether to include molecular descriptors. Default: True.

    Returns:
        pd.DataFrame: DataFrame with original columns plus:
            - 2048 Morgan fingerprint columns (Morgan_0 to Morgan_2047)
            - 7 general molecular descriptors (if include_descriptors=True)

    Notes:
        - Radius=2 (ECFP4) is standard for drug-like molecules
        - 2048 bits provides good balance between information and sparsity
        - Morgan fingerprints complement MACCS keys by capturing learned patterns
    """

    df = df.copy()  # Work on a copy to avoid modifying the original

    # --- 1. MOLECULAR DESCRIPTORS ---
    if include_descriptors:
        descriptor_funcs = {
            'MolecularWeight': lambda mol: Descriptors.MolWt(mol) if mol else np.nan,
            'LogP': lambda mol: Descriptors.MolLogP(mol) if mol else np.nan,
            'TPSA': lambda mol: Descriptors.TPSA(mol) if mol else np.nan,
            'NumRotatableBonds': lambda mol: Descriptors.NumRotatableBonds(mol) if mol else np.nan,
            'NumAromaticRings': lambda mol: Descriptors.NumAromaticRings(mol) if mol else np.nan,
            'NumHDonors': lambda mol: Lipinski.NumHDonors(mol) if mol else np.nan,
            'NumHAcceptors': lambda mol: Lipinski.NumHAcceptors(mol) if mol else np.nan,
        }

        print(f"Calculating {len(descriptor_funcs)} molecular descriptors...")

        for desc_name, desc_func in descriptor_funcs.items():
            df[desc_name] = df[mol_col].apply(desc_func)

    # --- 2. MORGAN FINGERPRINTS ---
    print(f"Generating Morgan fingerprints (radius={radius}, {n_bits} bits)...")

    def get_morgan_fingerprint(mol):
        """
        Generate Morgan (ECFP) fingerprint for a molecule.

        Args:
            mol: RDKit Mol object

        Returns:
            numpy array: Array of n_bits binary values (0 or 1)
        """
        if mol is None:
            return np.zeros(n_bits, dtype=int)

        try:
            # Generate Morgan fingerprint as bit vector
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            # Convert to numpy array
            arr = np.zeros((n_bits,), dtype=int)
            AllChem.DataStructs.ConvertToNumpyArray(morgan_fp, arr)
            return arr
        except Exception as e:
            print(f"Warning: Could not generate Morgan fingerprint: {e}")
            return np.zeros(n_bits, dtype=int)

    # Generate fingerprints for all molecules
    morgan_arrays = df[mol_col].apply(get_morgan_fingerprint).tolist()

    # Create column names for Morgan fingerprint bits
    morgan_col_names = [f'Morgan_{i}' for i in range(n_bits)]

    # Create DataFrame from fingerprint arrays
    df_morgan = pd.DataFrame(morgan_arrays, columns=morgan_col_names, index=df.index)

    # Concatenate Morgan features to main DataFrame
    df = pd.concat([df, df_morgan], axis=1)

    # --- 3. REORGANIZE COLUMNS ---
    # Move Molecule column to position 1 (after the first column, usually SMILES/ID)
    if mol_col in df.columns:
        col_to_move = df.pop(mol_col)
        df.insert(1, mol_col, col_to_move)

    # --- 4. SUMMARY REPORT ---
    num_descriptors = len(descriptor_funcs) if include_descriptors else 0
    num_morgan = n_bits
    total_features = num_descriptors + num_morgan

    print(f"\nMorgan Featurization Complete!")
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Features added:")
    print(f"  - Molecular descriptors: {num_descriptors}")
    print(f"  - Morgan fingerprint bits: {num_morgan}")
    print(f"  - Total features: {total_features}")
    print(f"  - Fingerprint type: ECFP{radius*2} (radius={radius})\n")

    return df
