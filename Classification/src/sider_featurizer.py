import pandas as pd
import numpy as np
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import rdFingerprintGenerator

def featurizer(df: pd.DataFrame, mol_col: str = 'Molecule', fpSize: int = 2048):
    """
    Calculates Morgan Fingerprints (2048 bits) and the specified RDKit descriptors
    using the MolecularDescriptorCalculator, adds them to the DataFrame, and cleans up.
    """

    # Descriptors List
    descriptors = [
        'MolWt',
        'TPSA',
        'MolLogP',
        'NumHAcceptors',
        'NumHDonors',
        'RingCount',
        'NumAromaticHeterocycles'
    ]

    calculator = MolecularDescriptorCalculator(descriptors)

    def calc_features(mol):
        if mol is None:
          return [np.nan] * len(descriptors)
        return list(calculator.CalcDescriptors(mol))

    # Apply the calculator to the 'Molecule' column
    properties_series = df[mol_col].apply(calc_features)

    # Convert the Series of lists into a new DataFrame with named columns
    df_simple_features = pd.DataFrame(properties_series.tolist(), columns=descriptors, index=df.index)

    df = pd.concat([df, df_simple_features], axis=1)

    # MORGAN FINGERPRINTS
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fpSize)

    df['FingerPrint_Obj'] = df[mol_col].apply(
        lambda mol: mfpgen.GetFingerprint(mol) if mol is not None else None
    )

    # Explode the BitVect column into 2048 binary columns (0s and 1s)
    # Create the list of 0s and 1s for each molecule
    fp_arrays = [
        fp.ToList() if fp is not None else [0] * fpSize
        for fp in df['FingerPrint_Obj']
    ]

    fp_names = [f'Morgan_{i}' for i in range(fpSize)]
    df_fp_features = pd.DataFrame(np.array(fp_arrays), columns=fp_names, index=df.index)

    # Concatenate the new FP features to the main DataFrame
    df = pd.concat([df, df_fp_features], axis=1)
    col_to_move = df.pop('Molecule')
    df.insert(1, 'Molecule', col_to_move)

    total_features = len(descriptors) + fpSize
    print(f"Feature engineering complete. Final DataFrame shape: {df.shape}")
    print(f"Total features added: {len(descriptors)} simple + {fpSize} Morgan = {total_features}")

    return df
