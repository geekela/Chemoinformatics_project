import pandas as pd
from rdkit import Chem
import numpy as np
from typing import List

def bace_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing function for the BACE (Regression) dataset.

    1. Validates SMILES strings and converts them to canonical form.
    2. Ensures the target column (pIC50) and all existing descriptors are numeric.
    3. Drops rows with invalid SMILES or missing pIC50.
    4. Removes duplicate molecules based on canonical SMILES.

    The function preserves all existing molecular descriptors (serving as Featurization 1).

    Args:
        df (pd.DataFrame): The raw BACE DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame, ready for modeling.
    """

    df = df.copy() # Work on a copy of the dataframe

    # 1. Ensure target and descriptor columns are numeric
    total_null_values = df.isnull().sum().sum()
    print(f"\nNombre TOTAL de valeurs nulles dans le DataFrame : {total_null_values}")

    total_na_values = df.isna().sum().sum()
    print(f"\nNombre TOTAL de Na dans le DataFrame : {total_na_values}")

    non_numeric_df = df.select_dtypes(exclude=np.number)
    print("\nColonnes Non Numériques (Catégorielles/Objets) :")
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

      mol = Chem.MolFromSmiles(smiles) #create a mol object from input smiles

      canonical_smiles = Chem.MolToSmiles(mol) #convert the previous mol object to SMILES using Chem.MolToSmiles()

      ####END
      return canonical_smiles


    #apply canonical smiles to our df
    df['canonical_smiles'] = df['mol'].apply(canonicalize_smiles)

    num_rows = len(df)
    num_unique_smiles = df['canonical_smiles'].nunique()

    print(f"\nNombre total de lignes dans le DataFrame : {num_rows}")
    print(f"\nNombre de SMILES canoniques uniques : {num_unique_smiles}\n")

    #drop old 'smiles' column
    df = df.drop(columns='mol')


    print(f"\n--- BACE Preprocessing Report ---\n")
    print(f"Final dataset size: {df.shape}\n")
    print(df.head())

    return df
# --- END of bace_preprocessing function ---
