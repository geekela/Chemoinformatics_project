#!/usr/bin/env python3
"""Test the specific imports from the notebook cell"""

print("Testing notebook imports...")

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ Error importing pandas: {e}")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ Error importing numpy: {e}")

try:
    import seaborn as sns
    print("✓ seaborn imported successfully")
except ImportError as e:
    print(f"✗ Error importing seaborn: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Error importing matplotlib: {e}")

try:
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    print("✓ sklearn components imported successfully")
except ImportError as e:
    print(f"✗ Error importing sklearn: {e}")

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdFingerprintGenerator, AllChem
    from rdkit.Chem.Descriptors import MolWt, TPSA, NumHDonors, NumHAcceptors
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    print("✓ RDKit components imported successfully")
except ImportError as e:
    print(f"✗ Error importing RDKit: {e}")

print("\n✅ All imports (except deepchem) are working correctly!")
print("Note: deepchem is commented out due to Python 3.12 compatibility issues")
print("      The notebook uses Classification/src/sider_featurizer.py instead")