"""
Complete Training Script: GNN for BACE Regression
==================================================

This script follows the reference code pattern (copia_di_intro_to_gnns.py)
adapted for the BACE dataset.

Task: Predict pIC50 values (BACE inhibition) - REGRESSION
Model: Message Passing Neural Network (MPNN)
Architecture: Identical to reference ESOL implementation
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from Regression.src.bace_gnn import BACEGraphDataset, MPNN_BACE


def main():
    print("=" * 80)
    print("GNN Training for BACE pIC50 Prediction (Regression)")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n STEP 1: Loading BACE dataset...")

    data_path = 'data/raw/bace.csv'

    try:
        df = pd.read_csv(data_path)
        print(f" Loaded {len(df)} molecules")
        print(f"   SMILES column: 'mol'")
        print(f"   Target column: 'pIC50'")
        print(f"   pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
        print(f"   pIC50 mean: {df['pIC50'].mean():.2f}, std: {df['pIC50'].std():.2f}")
    except FileNotFoundError:
        print(f" File not found: {data_path}")
        return
    except KeyError as e:
        print(f" Required column not found: {e}")
        print(f"   Available columns: {list(df.columns[:10])}...")
        return

    # =========================================================================
    # STEP 2: Create Graph Dataset
    # =========================================================================
    print("\n STEP 2: Converting molecules to graphs...")
    print("   This may take a few minutes on first run...")

    # Create graph dataset
    dataset = BACEGraphDataset(
        root='data/processed/bace_graphs',
        df=df,
        smiles_col='mol',
        target_col='pIC50'
    )

    print(f" Created {len(dataset)} molecular graphs")
    print(f"\n   Example graph (first molecule):")
    print(f"     • Atoms (nodes): {dataset[0].num_nodes}")
    print(f"     • Bonds (edges): {dataset[0].num_edges}")
    print(f"     • Node features: {dataset[0].x.shape}")
    print(f"     • Edge features: {dataset[0].edge_attr.shape}")
    print(f"     • Target (pIC50): {dataset[0].y.item():.2f}")

    # =========================================================================
    # STEP 3: Normalize Targets (CRITICAL for regression!)
    # =========================================================================
    print("\n STEP 3: Normalizing targets...")

    # Calculate statistics
    mean = dataset.data.y.mean()
    std = dataset.data.y.std()

    print(f"   Original pIC50 - Mean: {mean:.4f}, Std: {std:.4f}")

    # Normalize: (y - mean) / std
    dataset.data.y = (dataset.data.y - mean) / std

    print(f"   Normalized pIC50 - Mean: {dataset.data.y.mean():.4f}, Std: {dataset.data.y.std():.4f}")
    print("    Targets normalized to mean=0, std=1")

    # Convert to Python floats for model
    mean = mean.item()
    std = std.item()

    # =========================================================================
    # STEP 4: Split Dataset
    # =========================================================================
    print("\n  STEP 4: Splitting data...")

    # Shuffle dataset
    dataset = dataset.shuffle()

    # 70% train, 15% val, 15% test (same as reference)
    train_idx, temp_idx = train_test_split(
        range(len(dataset)),
        test_size=0.3,
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42
    )

    # Create subsets
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    print(f"   Training set: {len(train_dataset)} molecules")
    print(f"   Validation set: {len(val_dataset)} molecules")
    print(f"   Test set: {len(test_dataset)} molecules")

    # =========================================================================
    # STEP 5: Create Data Loaders
    # =========================================================================
    print("\n STEP 5: Creating data loaders...")

    # TODO(human): Adjust batch_size based on your hardware
    # - 32: Standard for 8GB+ GPU
    # - 16: Good for 4-8GB GPU
    # - 8: Safe for CPU or limited memory
    batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f" Data loaders created (batch size: {batch_size})")

    # =========================================================================
    # STEP 6: Initialize Model
    # =========================================================================
    print("\n STEP 6: Initializing GNN model...")

    # TODO(human): Experiment with these hyperparameters
    # - hidden_dim: 32, 64, 128, 256
    # - lr: 1e-4, 1e-3, 1e-2
    model = MPNN_BACE(
        hidden_dim=64,      # Embedding dimension
        out_dim=1,          # Single regression output
        std=std,            # For denormalization
        lr=1e-3            # Learning rate
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f" Model initialized")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Hidden dimension: 64")
    print(f"   Learning rate: 0.001")
    print(f"   Architecture: AtomEncoder → NNConv+GRU (×3) → GlobalAddPool → MLP")

    # =========================================================================
    # STEP 7: Setup Trainer
    # =========================================================================
    print("\n  STEP 7: Configuring trainer...")

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,        # Stop if no improvement for 15 epochs
        mode='min',
        verbose=True
    )

    checkpoint = ModelCheckpoint(
        monitor='val_rmse',
        mode='min',
        save_top_k=1,
        dirpath='models/checkpoints',
        filename='bace-gnn-best-{epoch:02d}-{val_rmse:.3f}'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,     # Maximum epochs (early stopping will likely stop earlier)
        callbacks=[early_stop, checkpoint],
        accelerator='auto', # Automatically uses MPS/CUDA/CPU
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=5,
        enable_model_summary=True
    )

    print(" Trainer configured")
    print(f"   Max epochs: 100")
    print(f"   Early stopping patience: 15")
    print(f"   Device: {trainer.accelerator}")

    # =========================================================================
    # STEP 8: Train!
    # =========================================================================
    print("\n STEP 8: Starting training...")
    print("   (This will take 10-30 minutes depending on your hardware)")
    print()

    trainer.fit(model, train_loader, val_loader)

    print("\n Training complete!")
    print(f"   Best model saved to: {checkpoint.best_model_path}")
    print(f"   Best validation RMSE: {checkpoint.best_model_score:.4f}")

    # =========================================================================
    # STEP 9: Evaluate on Test Set
    # =========================================================================
    print("\n STEP 9: Evaluating on test set...")

    # Load best model and test
    test_results = trainer.test(model, test_loader, ckpt_path='best')

    # =========================================================================
    # STEP 10: Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print(" BACE GNN TRAINING COMPLETE!")
    print("=" * 80)
    print("\nDataset Info:")
    print(f"  • Total molecules: {len(dataset)}")
    print(f"  • pIC50 range (original): {mean - 3*std:.2f} - {mean + 3*std:.2f}")
    print(f"  • Train/Val/Test split: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    print("\nModel Performance:")
    print(f"  • Test RMSE: {test_results[0]['test_rmse']:.4f}")
    print(f"  • Test MSE: {test_results[0]['test_mse']:.4f}")

    print("\nNext Steps:")
    print("  1. Compare with baseline models (RF, XGBoost, SVR)")
    print("  2. Try hyperparameter optimization:")
    print("     - Vary hidden_dim (32, 64, 128, 256)")
    print("     - Vary learning rate (1e-4, 1e-3, 1e-2)")
    print("     - Vary batch size (16, 32, 64)")
    print("  3. Analyze predictions vs actual values")
    print("  4. Visualize learned molecular representations")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
