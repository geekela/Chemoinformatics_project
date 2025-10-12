import pandas as pd
import numpy as np
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import rdFingerprintGenerator

def featurizer(df: pd.DataFrame, mol_col: str = 'Molecule', fpSize: int = 2048) -> pd.DataFrame:
    """
    Calculates Morgan Fingerprints (2048 bits) and the specified RDKit descriptors
    using the MolecularDescriptorCalculator, adds them to the DataFrame, and cleans up.

    Args:
        df (pd.DataFrame): DataFrame containing the RDKit Mol objects.
        mol_col (str): Name of the column containing the Mol objects ('Molecule').
        fpSize (int): Size of the Morgan fingerprint vector.

    Returns:
        pd.DataFrame: The single, fully featurized DataFrame (Targets + Features).
    """

    # --- 1. SETUP: Descriptors List ---
    # Using the standard descriptor names for the RDKit calculator
    descriptors = [
        'MolWt',
        'TPSA',
        'MolLogP',
        'NumHAcceptors',
        'NumHDonors',
        'RingCount',
        'NumAromaticHeterocycles'
    ]

    # Create the descriptor calculator object
    calculator = MolecularDescriptorCalculator(descriptors)

    # --- 2. CALCULATE AND ADD SIMPLE DESCRIPTORS
    def calc_features(mol):
        if mol is None:
          return [np.nan] * len(descriptors)
        return list(calculator.CalcDescriptors(mol))

    # Apply the calculator to the 'Molecule' column
    properties_series = df[mol_col].apply(calc_features)

    # Convert the Series of lists into a new DataFrame with named columns
    df_simple_features = pd.DataFrame(properties_series.tolist(), columns=descriptors, index=df.index)

    # Concatenate the simple features to the main DataFrame
    df = pd.concat([df, df_simple_features], axis=1)

    # --- 3. CALCULATE AND ADD MORGAN FINGERPRINTS ---

    # Create a fingerprint generator. This object is used to create any fingerprint from a specific familiy.
    # These parameters are commonly used.
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fpSize)

    # 3a. Generate BitVect objects (intermediate step)
    df['FingerPrint_Obj'] = df[mol_col].apply(
        lambda mol: mfpgen.GetFingerprint(mol) if mol is not None else None
    )

    # 3b. Explode the BitVect column into 2048 binary columns (0s and 1s)

    # Create the list of 0s and 1s for each molecule
    fp_arrays = [
        fp.ToList() if fp is not None else [0] * fpSize
        for fp in df['FingerPrint_Obj']
    ]

    # Create the feature names and a temporary DataFrame for FPs
    fp_names = [f'Morgan_{i}' for i in range(fpSize)]
    df_fp_features = pd.DataFrame(np.array(fp_arrays), columns=fp_names, index=df.index)

    # Concatenate the new FP features to the main DataFrame
    df = pd.concat([df, df_fp_features], axis=1)
    col_to_move = df.pop('Molecule')
    df.insert(1, 'Molecule', col_to_move)

    # --- 4. CLEANUP ---

    total_features = len(descriptors) + fpSize
    print(f"Feature engineering complete. Final DataFrame shape: {df.shape}")
    print(f"Total features added: {len(descriptors)} simple + {fpSize} Morgan = {total_features}")

    return df
