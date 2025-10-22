import pandas as pd
from rdkit import Chem
import numpy as np
from typing import List

def sider_preprocessing(df: pd.DataFrame):

    df = df.copy() # Work on a copy of the dataframe

    # Ensure there is no NA and Null
    total_null_values = df.isnull().sum().sum()
    if total_null_values > 0:
      print(f"\nTOTAL number of null values in the DataFrame: {total_null_values}")

    total_na_values = df.isna().sum().sum()
    if total_null_values > 0:
      print(f"\nTOTAL number of Na in the DataFrame: {total_na_values}")

    # Ensure the 27 labels are numeric
    non_numeric_df = df.select_dtypes(exclude=np.number)
    print("\nNon-Numeric Columns (Categorical/Objects):")
    print(non_numeric_df.columns.tolist())


    # Validate and Canonicalize SMILES
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

        return canonical_smiles


    # apply canonical smiles to our df
    df['canonical_smiles'] = df['smiles'].apply(canonicalize_smiles)

    num_rows = len(df)
    num_unique_smiles = df['canonical_smiles'].nunique()
    if num_rows - num_unique_smiles != 0:
      print(f"\nTotal number of rows in the DataFrame: {num_rows}")
      print(f"\nNumber of unique canonical SMILES: {num_unique_smiles}\n")

    # Replace old smiles column by the canonalized smiles
    df = df.drop(columns='smiles')

    last_col_name = df.columns[-1]
    col_to_move = df.pop(last_col_name)
    df.insert(0, last_col_name, col_to_move)

    # Create molecule column without using PandasTools for better compatibility
    df['Molecule'] = df['canonical_smiles'].apply(lambda x: Chem.MolFromSmiles(x))

    print(f"\n--- SIDER Preprocessing Report ---\n")

    print(f"Final dataset size: {df.shape}\n")
    # display(df)  # Only works in Jupyter

    return df
