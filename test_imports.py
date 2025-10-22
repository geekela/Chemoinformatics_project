#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, '/Users/elisa/CHEMO/Chemoinformatics_project')

def test_imports():
    """Test all the imports from the notebook"""

    print("Testing standard library imports...")
    try:
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        print("✓ Standard libraries imported successfully")
    except ImportError as e:
        print(f"✗ Error importing standard libraries: {e}")
        return False

    print("\nTesting deepchem import...")
    try:
        import deepchem as dc
        print("✓ Deepchem imported successfully")
    except ImportError as e:
        print(f"✗ Error importing deepchem: {e}")
        print("  Note: DeepChem may have compatibility issues with Python 3.12")

    print("\nTesting sklearn imports...")
    try:
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Error importing sklearn: {e}")
        return False

    print("\nTesting RDKit imports...")
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, rdFingerprintGenerator, AllChem
        from rdkit.Chem.Descriptors import MolWt, TPSA, NumHDonors, NumHAcceptors
        from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
        print("✓ RDKit imported successfully")
    except ImportError as e:
        print(f"✗ Error importing RDKit: {e}")
        return False

    print("\nTesting custom module imports...")
    try:
        from Classification.src.sider_preprocessing import sider_preprocessing
        print("✓ sider_preprocessing imported successfully")
    except ImportError as e:
        print(f"✗ Error importing sider_preprocessing: {e}")
        return False

    try:
        from Classification.src.sider_featurizer import featurizer
        print("✓ sider_featurizer imported successfully")
    except ImportError as e:
        print(f"✗ Error importing sider_featurizer: {e}")
        return False

    try:
        from Classification.src.sider_pca import analyze_pca_variance
        print("✓ sider_pca imported successfully")
    except ImportError as e:
        print(f"✗ Error importing sider_pca: {e}")
        return False

    try:
        from Classification.src.sider_baseline_models import train_and_evaluate_models
        print("✓ sider_baseline_models imported successfully")
    except ImportError as e:
        print(f"✗ Error importing sider_baseline_models: {e}")
        return False

    try:
        from Classification.src.sider_umap import plot_umap
        print("✓ sider_umap imported successfully")
    except ImportError as e:
        print(f"✗ Error importing sider_umap: {e}")
        return False

    print("\nTesting PyTorch and PyTorch Geometric imports...")
    try:
        import torch
        from torch_geometric.loader import DataLoader
        import pytorch_lightning as pl
        print("✓ PyTorch and PyTorch Geometric imported successfully")
    except ImportError as e:
        print(f"✗ Error importing PyTorch/PyTorch Geometric: {e}")
        return False

    try:
        from Classification.src.sider_gnn import SIDERGraphDataset, MPNN_SIDER
        print("✓ sider_gnn imported successfully")
    except ImportError as e:
        print(f"✗ Error importing sider_gnn: {e}")
        return False

    try:
        from Classification.src.sider_gnn_bayesian_opti import objective
        print("✓ sider_gnn_bayesian_opti imported successfully")
    except ImportError as e:
        print(f"✗ Error importing sider_gnn_bayesian_opti: {e}")
        return False

    try:
        from Classification.src.sider_svm_optimization import optimize_svm
        print("✓ sider_svm_optimization imported successfully")
    except ImportError as e:
        print(f"✗ Error importing sider_svm_optimization: {e}")
        return False

    print("\nTesting other ML library imports...")
    try:
        import optuna
        print("✓ Optuna imported successfully")
    except ImportError as e:
        print(f"✗ Error importing optuna: {e}")
        return False

    try:
        import umap.umap_ as umap
        print("✓ UMAP imported successfully")
    except ImportError as e:
        print(f"✗ Error importing umap: {e}")
        return False

    try:
        from transformers import AutoTokenizer, AutoModel
        print("✓ Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Error importing transformers: {e}")
        print("  You may need to install: pip install transformers")

    print("\n" + "="*50)
    print("Import test completed!")
    return True

if __name__ == "__main__":
    test_imports()