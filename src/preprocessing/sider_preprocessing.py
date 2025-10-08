import pandas as pd
from rdkit import Chem
import numpy as np
from typing import List

def sider_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing function for the SIDER (Classification) dataset.

    Args:
        df (pd.DataFrame): The raw SIDER DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame, ready for modeling.
    """

    df = df.copy() # Work on a copy of the dataframe

    # 1. Ensure target and descriptor columns are numeric
    total_null_values = df.isnull().sum().sum()
    print(f"\nTOTAL number of null values in the DataFrame: {total_null_values}")

    total_na_values = df.isna().sum().sum()
    print(f"\nTOTAL number of Na in the DataFrame: {total_na_values}")

    non_numeric_df = df.select_dtypes(exclude=np.number)
    print("\nNon-Numeric Columns (Categorical/Objects):")
    print(non_numeric_df.columns.tolist())


    # 3. Validate and Canonicalize SMILES
    def canonicalize_smiles(smiles):
        '''This function takes a non-canonical SMILES and
        returns the canonical version

        Args:
            -smiles: str, non-canonical SMILES of a molecule

        Out:
            - canonical_smiles: str, canonical SMILES of the molecule
        '''

        mol = Chem.MolFromSmiles(smiles) # create a mol object from input smiles

        canonical_smiles = Chem.MolToSmiles(mol) # convert the previous mol object to SMILES using Chem.MolToSmiles()

        ####END
        return canonical_smiles


    # apply canonical smiles to our df
    df['canonical_smiles'] = df['smiles'].apply(canonicalize_smiles)

    num_rows = len(df)
    num_unique_smiles = df['canonical_smiles'].nunique()

    print(f"\nTotal number of rows in the DataFrame: {num_rows}")
    print(f"\nNumber of unique canonical SMILES: {num_unique_smiles}\n")

    # drop old 'smiles' column
    df = df.drop(columns='smiles')


    print(f"\n--- SIDER Preprocessing Report ---\n")
    print(f"Final dataset size: {df.shape}\n")
    print(df.head())

    return df
# --- END of sider_preprocessing function ---
