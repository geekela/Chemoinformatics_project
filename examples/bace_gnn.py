"""
BACE GNN: Complete Pipeline for Graph Neural Network Molecular Property Prediction

This consolidated script includes all functionality for:
1. Molecular featurization (fingerprints and descriptors)
2. Graph dataset creation for GNN
3. GNN model architecture (Message Passing Neural Network)
4. Model training with PyTorch Lightning
5. Baseline model comparison (RF, XGBoost, SVR)
6. Comprehensive evaluation and visualization
7. Error analysis and molecular property correlation

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Standard library
import argparse
import glob
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

# Data manipulation
import numpy as np
import pandas as pd

# Chemistry and molecular processing
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Crippen, Lipinski, rdMolDescriptors

# Machine learning - scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Machine learning - XGBoost
import xgboost as xgb

# Deep learning - PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, GRU
from torch_geometric.nn import NNConv, global_add_pool
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Deep learning - PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Some visualizations will be unavailable.")

# Settings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def generate_ecfp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """
    Generate Extended Connectivity Fingerprints (ECFP/Morgan fingerprints).

    Args:
        smiles: SMILES string representation of molecule
        radius: Fingerprint radius (default: 2 for ECFP4)
        n_bits: Number of bits in fingerprint (default: 1024)

    Returns:
        numpy array of fingerprint bits
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def generate_maccs(smiles: str) -> np.ndarray:
    """
    Generate MACCS keys fingerprints (166 structural keys).

    Args:
        smiles: SMILES string representation of molecule

    Returns:
        numpy array of MACCS keys (167 bits)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def generate_rdkit_descriptors(smiles: str) -> np.ndarray:
    """
    Generate comprehensive RDKit 2D molecular descriptors.

    Args:
        smiles: SMILES string representation of molecule

    Returns:
        numpy array of molecular descriptors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(200)  # Placeholder for failed molecules

    descriptors = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            descriptors.append(val)
        except:
            descriptors.append(0.0)

    return np.array(descriptors)


def calculate_molecular_properties(smiles: str) -> Dict[str, float]:
    """
    Calculate key molecular properties for error analysis.

    Args:
        smiles: SMILES string representation of molecule

    Returns:
        Dictionary of molecular properties
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
        'NumAromaticRings': Lipinski.NumAromaticRings(mol),
        'NumRings': Lipinski.RingCount(mol),
        'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
        'FractionCsp3': Lipinski.FractionCsp3(mol)
    }


def featurize_molecules_simple(df: pd.DataFrame,
                              smiles_col: str = 'canonical_smiles',
                              target_col: str = 'pIC50',
                              methods: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate multiple types of molecular features for baseline models.

    Args:
        df: DataFrame with SMILES and target values
        smiles_col: Column name for SMILES strings
        target_col: Column name for target variable
        methods: List of featurization methods ('ecfp', 'maccs', 'rdkit')

    Returns:
        Dictionary mapping method names to feature DataFrames
    """
    if methods is None:
        methods = ['ecfp', 'maccs', 'rdkit']

    smiles_list = df[smiles_col].tolist()
    target_list = df[target_col].tolist()

    features_dict = {}

    for method in methods:
        print(f"\n   Generating {method.upper()} features...")

        if method == 'ecfp':
            features = np.array([generate_ecfp(s) for s in smiles_list])
        elif method == 'maccs':
            features = np.array([generate_maccs(s) for s in smiles_list])
        elif method == 'rdkit':
            features = np.array([generate_rdkit_descriptors(s) for s in smiles_list])
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"     Shape before cleaning: {features.shape}")

        # Remove NaN columns
        nan_mask = ~np.isnan(features).any(axis=0)
        features = features[:, nan_mask]

        # Remove zero variance features
        if features.shape[1] > 0:
            selector = VarianceThreshold(threshold=0.0)
            features = selector.fit_transform(features)

        print(f"     Shape after cleaning: {features.shape}")

        # Create DataFrame
        feature_names = [f"{method}_{i}" for i in range(features.shape[1])]
        df_feat = pd.DataFrame(features, columns=feature_names)

        # Add SMILES and target
        df_feat.insert(0, smiles_col, df[smiles_col].values)
        df_feat.insert(1, target_col, target_list)

        features_dict[method] = df_feat

    return features_dict


class BACEGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for BACE molecular graphs.
    Converts SMILES strings to graph representations for GNN processing.
    """

    def __init__(self, root: str, df: pd.DataFrame,
                 smiles_col: str = 'mol',
                 target_col: str = 'pIC50',
                 transform=None, pre_transform=None, pre_filter=None):
        """
        Initialize BACE graph dataset.

        Args:
            root: Root directory for processed data
            df: DataFrame containing SMILES and targets
            smiles_col: Column name for SMILES strings
            target_col: Column name for target values
        """
        self.df = df
        self.smiles_col = smiles_col
        self.target_col = target_col
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.df))]

    def download(self):
        pass

    def process(self):
        for idx, row in self.df.iterrows():
            smiles = row[self.smiles_col]
            target = row[self.target_col]

            # Convert SMILES to graph
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Node features (atoms)
            atom_features = []
            for atom in mol.GetAtoms():
                features = self.get_atom_features(atom)
                atom_features.append(features)

            # Edge features (bonds)
            edge_indices = []
            edge_attrs = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_indices.extend([[i, j], [j, i]])  # Bidirectional
                bond_features = self.get_bond_features(bond)
                edge_attrs.extend([bond_features, bond_features])

            # Create PyTorch Geometric Data object
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            y = torch.tensor([target], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def get_atom_features(self, atom):
        """Extract atom-level features."""
        features = []
        features.append(atom.GetAtomicNum())
        features.append(atom.GetDegree())
        features.append(atom.GetFormalCharge())
        features.append(int(atom.GetHybridization()))
        features.append(int(atom.GetIsAromatic()))
        features.append(atom.GetTotalNumHs())
        features.append(int(atom.IsInRing()))
        features.append(int(atom.HasProp('_ChiralityPossible')))
        return features

    def get_bond_features(self, bond):
        """Extract bond-level features."""
        bt = bond.GetBondType()
        features = [
            bt == Chem.BondType.SINGLE,
            bt == Chem.BondType.DOUBLE,
            bt == Chem.BondType.TRIPLE,
            bt == Chem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
        return features

    def len(self):
        return len(self.df)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data


class MPNN_BACE(pl.LightningModule):
    """
    Message Passing Neural Network for BACE pIC50 prediction.
    Uses NNConv layers with GRU update for molecular property prediction.
    """

    def __init__(self, num_features: int = 8, num_edge_features: int = 6,
                 hidden_dim: int = 64, output_dim: int = 1,
                 dropout: float = 0.1, lr: float = 0.001,
                 mean: float = 0.0, std: float = 1.0):
        """
        Initialize MPNN model.

        Args:
            num_features: Number of atom features
            num_edge_features: Number of bond features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for regression)
            dropout: Dropout rate
            lr: Learning rate
            mean: Target mean for denormalization
            std: Target std for denormalization
        """
        super().__init__()
        self.save_hyperparameters()

        # Atom embedding
        self.atom_emb = nn.Linear(num_features, hidden_dim)

        # Bond embedding network for NNConv
        nn_layer = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        # Message passing layers
        self.conv = NNConv(hidden_dim, hidden_dim, nn_layer, aggr='add')
        self.gru = GRU(hidden_dim, hidden_dim)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Normalization parameters
        self.mean = mean
        self.std = std
        self.lr = lr

    def forward(self, batch):
        """Forward pass through the network."""
        # Initial atom embeddings
        x = self.atom_emb(batch.x)
        h = x.unsqueeze(0)

        # Bond embeddings
        edge_attr = batch.edge_attr

        # Message passing (3 iterations)
        for i in range(3):
            m = torch.relu(self.conv(x, batch.edge_index, edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = x.squeeze(0)

        # Graph-level pooling
        x = global_add_pool(x, batch.batch)

        # Final MLP
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze()

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning."""
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)

        # Denormalize for metrics
        pred_denorm = pred * self.std + self.mean
        y_denorm = batch.y * self.std + self.mean

        rmse = torch.sqrt(F.mse_loss(pred_denorm, y_denorm))
        mae = F.l1_loss(pred_denorm, y_denorm)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step for PyTorch Lightning."""
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)

        # Denormalize for metrics
        pred_denorm = pred * self.std + self.mean
        y_denorm = batch.y * self.std + self.mean

        rmse = torch.sqrt(F.mse_loss(pred_denorm, y_denorm))
        mae = F.l1_loss(pred_denorm, y_denorm)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_rmse', rmse, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)

        return {'loss': loss, 'rmse': rmse, 'mae': mae}

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        model_type: str = 'rf') -> Tuple[Any, Dict[str, float]]:
    """
    Train and evaluate a baseline model.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_type: Type of model ('rf', 'xgb', 'svr')

    Returns:
        Trained model and metrics dictionary
    """
    # Scale features for SVR
    if model_type == 'svr':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Initialize model
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    elif model_type == 'svr':
        model = SVR(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            epsilon=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }

    return model, metrics


def run_baseline_experiments(features_dict: Dict[str, pd.DataFrame],
                           test_size: float = 0.2,
                           random_state: int = 42) -> pd.DataFrame:
    """
    Run all baseline models with all featurization methods.

    Args:
        features_dict: Dictionary of featurized datasets
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        DataFrame with results for all model-feature combinations
    """
    results = []

    model_types = ['rf', 'xgb', 'svr']
    model_names = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'svr': 'SVR'
    }

    for feat_name, feat_df in features_dict.items():
        print(f"\nTesting models with {feat_name.upper()} features...")

        # Prepare data
        feature_cols = [c for c in feat_df.columns
                       if c not in ['canonical_smiles', 'mol', 'pIC50']]
        X = feat_df[feature_cols].values
        y = feat_df['pIC50'].values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        for model_type in model_types:
            print(f"  Training {model_names[model_type]}...")
            model, metrics = train_baseline_model(
                X_train, y_train, X_test, y_test, model_type
            )

            results.append({
                'Model': model_names[model_type],
                'Featurizer': feat_name.upper(),
                'RMSE': metrics['test_rmse'],
                'MAE': metrics['test_mae'],
                'R2': metrics['test_r2'],
                'Train_RMSE': metrics['train_rmse']
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')

    return results_df


def train_gnn(dataset: BACEGraphDataset,
              max_epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              patience: int = 20) -> MPNN_BACE:
    """
    Train the GNN model using PyTorch Lightning.

    Args:
        dataset: Graph dataset
        max_epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        patience: Early stopping patience

    Returns:
        Trained GNN model
    """
    # Split dataset
    num_samples = len(dataset)
    indices = torch.randperm(num_samples)

    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)

    train_dataset = dataset[indices[:train_size]]
    val_dataset = dataset[indices[train_size:train_size + val_size]]
    test_dataset = dataset[indices[train_size + val_size:]]

    # Create data loaders
    train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size)
    test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size)

    # Calculate normalization stats
    y_values = [dataset[i].y.item() for i in range(len(dataset))]
    mean = np.mean(y_values)
    std = np.std(y_values)

    # Initialize model
    model = MPNN_BACE(
        num_features=8,
        num_edge_features=6,
        hidden_dim=64,
        lr=learning_rate,
        mean=mean,
        std=std
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints',
        filename='bace-gnn-{epoch:02d}-{val_rmse:.4f}',
        save_top_k=1,
        monitor='val_rmse',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=TensorBoardLogger('logs/', name='bace_gnn'),
        enable_progress_bar=True,
        deterministic=True
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_loader)

    return model


def plot_model_comparison(results_df: pd.DataFrame, gnn_rmse: float,
                         save_path: str = 'results/model_comparison.png'):
    """
    Create bar plot comparing all models.

    Args:
        results_df: DataFrame with baseline results
        gnn_rmse: GNN test RMSE
        save_path: Path to save figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for plotting
    featurizers = results_df['Featurizer'].unique()
    models = results_df['Model'].unique()

    x = np.arange(len(featurizers))
    width = 0.25

    # Plot baseline models
    for i, model in enumerate(models):
        model_data = results_df[results_df['Model'] == model]
        rmse_values = [model_data[model_data['Featurizer'] == f]['RMSE'].values[0]
                      for f in featurizers]
        ax.bar(x + i*width, rmse_values, width, label=model, alpha=0.8)

    # Add GNN line
    ax.axhline(y=gnn_rmse, color='red', linestyle='--', linewidth=2, label='GNN (MPNN)')

    # Formatting
    ax.set_xlabel('Featurizer', fontweight='bold', fontsize=12)
    ax.set_ylabel('RMSE (pIC50 units)', fontweight='bold', fontsize=12)
    ax.set_title('BACE pIC50 Prediction: Model Performance Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(featurizers, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {save_path}")
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                    title: str, save_path: str):
    """
    Create scatter plot of predictions vs actual values.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    # Metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Add metrics text
    textstr = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # Formatting
    ax.set_xlabel('Actual pIC50', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted pIC50', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved prediction plot to: {save_path}")
    plt.close()


def analyze_prediction_errors(df: pd.DataFrame, predictions: np.ndarray,
                             save_path: str = 'results/error_analysis.png'):
    """
    Analyze and visualize prediction errors vs molecular properties.

    Args:
        df: DataFrame with molecules and properties
        predictions: Model predictions
        save_path: Path to save figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Calculate errors
    errors = np.abs(df['pIC50'].values - predictions)

    # Calculate molecular properties
    properties = []
    for smiles in df['mol'].values:
        props = calculate_molecular_properties(smiles)
        if props:
            properties.append(props)
        else:
            properties.append({k: np.nan for k in ['MW', 'LogP', 'TPSA', 'NumRotatableBonds']})

    props_df = pd.DataFrame(properties)
    props_df['AbsError'] = errors

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Error vs molecular properties
    properties_to_plot = ['MW', 'LogP', 'TPSA', 'NumRotatableBonds']

    for ax, prop in zip(axes.flat, properties_to_plot):
        ax.scatter(props_df[prop], props_df['AbsError'], alpha=0.6, s=30)
        ax.set_xlabel(prop, fontweight='bold')
        ax.set_ylabel('Absolute Error', fontweight='bold')
        ax.set_title(f'Prediction Error vs {prop}')
        ax.grid(alpha=0.3)

        # Add trend line
        mask = ~np.isnan(props_df[prop]) & ~np.isnan(props_df['AbsError'])
        if mask.sum() > 10:
            z = np.polyfit(props_df[prop][mask], props_df['AbsError'][mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(props_df[prop][mask].min(), props_df[prop][mask].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.5)

    plt.suptitle('Prediction Error Analysis by Molecular Properties', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved error analysis to: {save_path}")
    plt.close()


def main():
    """
    Main execution function - runs complete BACE analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='BACE GNN Complete Pipeline')
    parser.add_argument('--data-path', type=str, default='data/raw/bace.csv',
                       help='Path to BACE dataset CSV')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'all'],
                       help='Execution mode')
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='Maximum training epochs for GNN')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for GNN training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for GNN')

    args = parser.parse_args()

    print("BACE GNN - Complete Molecular Property Prediction Pipeline")

    print("\nSTEP 1: Loading BACE dataset...")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} molecules")
    print(f"Target range: {df['pIC50'].min():.2f} to {df['pIC50'].max():.2f}")

    if args.mode in ['train', 'all']:
        print("\nSTEP 2: Generating molecular features for baseline models...")

        # Rename column for compatibility
        df_temp = df.rename(columns={'mol': 'canonical_smiles'})
        features_dict = featurize_molecules_simple(
            df_temp,
            smiles_col='canonical_smiles',
            target_col='pIC50',
            methods=['ecfp', 'maccs', 'rdkit']
        )

        print(f"\nGenerated {len(features_dict)} feature sets:")
        for feat_name, feat_df in features_dict.items():
            n_features = len([c for c in feat_df.columns
                            if c not in ['canonical_smiles', 'pIC50']])
            print(f"  - {feat_name.upper()}: {n_features} features")

    if args.mode in ['train', 'all']:
        print("\nSTEP 3: Training baseline models (RF, XGBoost, SVR)...")
        baseline_results = run_baseline_experiments(features_dict)

        print("\nBaseline Results (sorted by RMSE):")
        print(baseline_results[['Model', 'Featurizer', 'RMSE', 'MAE', 'R2']].to_string(index=False))

        # Save results
        os.makedirs('results', exist_ok=True)
        baseline_results.to_csv('results/baseline_results.csv', index=False)
        print("\nSaved baseline results to: results/baseline_results.csv")

    print("\nSTEP 4: Creating graph dataset for GNN...")
    dataset = BACEGraphDataset(
        root='data/processed/bace_graphs',
        df=df,
        smiles_col='mol',
        target_col='pIC50'
    )
    print(f"Created graph dataset with {len(dataset)} samples")

    if args.mode in ['train', 'all']:
        print("\nSTEP 5: Training Graph Neural Network (MPNN)...")
        print(f"Configuration: {args.max_epochs} epochs, batch size {args.batch_size}, LR {args.learning_rate}")

        gnn_model = train_gnn(
            dataset,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        print("\nGNN training complete!")

    if args.mode in ['evaluate', 'all']:
        print("\nSTEP 6: Evaluating models and generating visualizations...")

        # Load best GNN checkpoint
        checkpoints = glob.glob('models/checkpoints/bace-gnn-*.ckpt')
        if checkpoints:
            best_checkpoint = sorted(checkpoints)[-1]
            print(f"Loading GNN from: {best_checkpoint}")

            # Load model
            gnn_model = MPNN_BACE.load_from_checkpoint(best_checkpoint)
            gnn_model.eval()

            # Get predictions
            loader = GeometricDataLoader(dataset, batch_size=32, shuffle=False)
            predictions = []
            actuals = []

            with torch.no_grad():
                for batch in loader:
                    pred = gnn_model(batch)
                    pred_denorm = pred * gnn_model.std + gnn_model.mean
                    actual_denorm = batch.y * gnn_model.std + gnn_model.mean
                    predictions.extend(pred_denorm.cpu().numpy())
                    actuals.extend(actual_denorm.cpu().numpy())

            predictions = np.array(predictions)
            actuals = np.array(actuals)

            # Calculate GNN metrics
            gnn_rmse = np.sqrt(mean_squared_error(actuals, predictions))
            gnn_mae = mean_absolute_error(actuals, predictions)
            gnn_r2 = r2_score(actuals, predictions)

            print(f"\nGNN Performance:")
            print(f"  RMSE: {gnn_rmse:.4f}")
            print(f"  MAE:  {gnn_mae:.4f}")
            print(f"  R²:   {gnn_r2:.4f}")

            # Generate visualizations
            if args.mode == 'all' and 'baseline_results' in locals():
                plot_model_comparison(baseline_results, gnn_rmse)

            plot_predictions(actuals, predictions,
                           'GNN (MPNN) Predictions vs Actual',
                           'results/gnn_predictions.png')

            analyze_prediction_errors(df, predictions)
        else:
            print("No GNN checkpoint found. Please train the model first.")

    print("\nANALYSIS COMPLETE")
    print("\nGenerated files:")
    if os.path.exists('results'):
        for file in os.listdir('results'):
            print(f"  - results/{file}")

    print("\nPipeline execution finished successfully!")


if __name__ == "__main__":
    main()