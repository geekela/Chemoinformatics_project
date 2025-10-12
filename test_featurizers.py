"""
Quick test script for MACCS and Morgan featurizers
"""
import pandas as pd
import numpy as np
from rdkit import Chem
import sys

# Add src to path
sys.path.append('src')

from featurization.sider_maccs_featurizer import featurizer_maccs
from featurization.sider_morgan_featurizer import featurizer_morgan

print("SIDER Featurization Test: MACCS vs Morgan")

# Load cleaned SIDER dataset
print("\n1. Loading SIDER dataset...")
df = pd.read_csv('data/processed/sider_cleaned_f1.csv')
print(f"   Dataset shape: {df.shape}")

# Create RDKit Molecule objects
print("\n2. Converting SMILES to RDKit molecules...")
df['Molecule'] = df['canonical_smiles'].apply(Chem.MolFromSmiles)
invalid_count = df['Molecule'].isna().sum()
print(f"   Invalid molecules: {invalid_count}")

if invalid_count > 0:
    print("   Removing invalid molecules...")
    df = df.dropna(subset=['Molecule'])

print(f"   Final dataset size: {len(df)} molecules")

# Test MACCS featurization
print("3. Testing MACCS Keys Featurizer")
df_maccs = featurizer_maccs(df.copy(), mol_col='Molecule', include_lipinski=True)
print(f"\nMACCS featurized shape: {df_maccs.shape}")

# Extract MACCS fingerprint columns
maccs_fps = df_maccs[[col for col in df_maccs.columns if 'MACCS' in col]]
print(f"MACCS fingerprint columns: {maccs_fps.shape[1]}")
print(f"Avg bits set per molecule: {maccs_fps.sum(axis=1).mean():.2f}")
print(f"Sparsity (% zeros): {(maccs_fps == 0).sum().sum() / maccs_fps.size * 100:.2f}%")

# Test Morgan featurization
print("4. Testing Morgan Fingerprint Featurizer")
df_morgan = featurizer_morgan(
    df.copy(),
    mol_col='Molecule',
    radius=2,
    n_bits=2048,
    include_descriptors=True
)
print(f"\nMorgan featurized shape: {df_morgan.shape}")

# Extract Morgan fingerprint columns
morgan_fps = df_morgan[[col for col in df_morgan.columns if 'Morgan' in col]]
print(f"Morgan fingerprint columns: {morgan_fps.shape[1]}")
print(f"Avg bits set per molecule: {morgan_fps.sum(axis=1).mean():.2f}")
print(f"Sparsity (% zeros): {(morgan_fps == 0).sum().sum() / morgan_fps.size * 100:.2f}%")

# Comparison
print("5. Feature Comparison Summary")
print(f"{'Metric':<35} {'MACCS':<15} {'Morgan'}")
print("-"*70)
print(f"{'Number of fingerprint bits':<35} {maccs_fps.shape[1]:<15} {morgan_fps.shape[1]}")
print(f"{'Avg bits set per molecule':<35} {maccs_fps.sum(axis=1).mean():<15.2f} {morgan_fps.sum(axis=1).mean():.2f}")
print(f"{'Sparsity (% zeros)':<35} {(maccs_fps == 0).sum().sum() / maccs_fps.size * 100:<15.2f} {(morgan_fps == 0).sum().sum() / morgan_fps.size * 100:.2f}")
print(f"{'Density (% ones)':<35} {(maccs_fps == 1).sum().sum() / maccs_fps.size * 100:<15.2f} {(morgan_fps == 1).sum().sum() / morgan_fps.size * 100:.2f}")

# Save featurized datasets
print("6. Saving Featurized Datasets")

df_maccs_save = df_maccs.drop(columns=['Molecule'])
df_morgan_save = df_morgan.drop(columns=['Molecule'])

df_maccs_save.to_csv('data/processed/sider_featurized_maccs.csv', index=False)
df_morgan_save.to_csv('data/processed/sider_featurized_morgan.csv', index=False)

print("MACCS saved to: data/processed/sider_featurized_maccs.csv")
print(f"   Shape: {df_maccs_save.shape}")
print("Morgan saved to: data/processed/sider_featurized_morgan.csv")
print(f"   Shape: {df_morgan_save.shape}")

print("Featurization Test Complete!")
print("\nBoth MACCS and Morgan fingerprints are ready for modeling!")
print("Next steps:")
print("  1. Build Baseline Model (Random Forest)")
print("  2. Build Model 2 (XGBoost with tuning)")
print("  3. Build Model 3 (Neural Network or ensemble)")
print("  4. Compare MACCS vs Morgan performance across all models")
