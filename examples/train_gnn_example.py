"""
Example: Training a GNN for Molecular Property Prediction
===========================================================

This script demonstrates the complete workflow for training a Graph Neural Network
on molecular data using your new Classification module.

Prerequisites:
- Preprocessed SIDER data with canonical SMILES
- PyTorch Geometric and dependencies installed
"""

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from Classification.src.sider_gnn import SIDERGraphDataset, MPNN_SIDER


def main():
    print("=" * 70)
    print("GNN Training Example for Molecular Property Prediction")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Load and Prepare Data
    # =========================================================================
    print("\n STEP 1: Loading data...")

    # TODO(human): Update this path to your actual data file
    # The CSV should have:
    # - 'canonical_smiles' column
    # - 27 columns for side effect labels (binary 0/1)
    data_path = 'data/processed/sider_cleaned.csv'

    try:
        df = pd.read_csv(data_path)
        print(f" Loaded {len(df)} molecules")
        print(f"   Columns: {list(df.columns[:5])}...")
    except FileNotFoundError:
        print(f" File not found: {data_path}")
        print("   Please run preprocessing first or update the path")
        return

    # =========================================================================
    # STEP 2: Create Graph Dataset
    # =========================================================================
    print("\n STEP 2: Converting molecules to graphs...")
    print("   This may take a few minutes for the first time...")

    # This will save processed graphs to disk for faster loading next time
    dataset = SIDERGraphDataset(
        root='data/processed/sider_graphs',
        df=df
    )

    print(f" Created {len(dataset)} molecular graphs")
    print(f"\n   Example graph (first molecule):")
    print(f"     • Atoms (nodes): {dataset[0].num_nodes}")
    print(f"     • Bonds (edges): {dataset[0].num_edges}")
    print(f"     • Node features: {dataset[0].x.shape}")
    print(f"     • Edge features: {dataset[0].edge_attr.shape}")
    print(f"     • Labels: {dataset[0].y.shape}")

    # =========================================================================
    # STEP 3: Train/Validation/Test Split
    # =========================================================================
    print("\n  STEP 3: Splitting data...")

    # 70% train, 15% validation, 15% test
    train_idx, temp_idx = train_test_split(
        range(len(dataset)),
        test_size=0.3,
        random_state=42,
        shuffle=True
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42
    )

    print(f"   Training set: {len(train_idx)} molecules")
    print(f"   Validation set: {len(val_idx)} molecules")
    print(f"   Test set: {len(test_idx)} molecules")

    # Create data loaders
    # TODO(human): Adjust batch_size based on your GPU memory
    # - 32: Good for 8GB+ GPU
    # - 16: Good for 4-8GB GPU
    # - 8: Safe for CPU or limited memory
    batch_size = 32

    train_loader = DataLoader(
        [dataset[i] for i in train_idx],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging, increase for speed
    )
    val_loader = DataLoader(
        [dataset[i] for i in val_idx],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        [dataset[i] for i in test_idx],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f" Created data loaders (batch size: {batch_size})")

    # =========================================================================
    # STEP 4: Initialize Model
    # =========================================================================
    print("\n STEP 4: Initializing GNN model...")

    # TODO(human): Experiment with these hyperparameters
    # See GNN_Tutorial.md for guidance on choosing values
    model = MPNN_SIDER(
        hidden_dim=128,    # Embedding dimension (try 64, 128, 256)
        out_dim=27,        # Number of side effects (don't change)
        lr=1e-3           # Learning rate (try 1e-4, 1e-3, 1e-2)
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f" Model initialized")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Hidden dimension: 128")
    print(f"   Learning rate: 0.001")

    # =========================================================================
    # STEP 5: Setup Training
    # =========================================================================
    print("\n  STEP 5: Configuring trainer...")

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,        # Stop if no improvement for 10 epochs
        mode='min',
        verbose=True
    )

    checkpoint = ModelCheckpoint(
        monitor='val_roc_auc',
        mode='max',
        save_top_k=1,
        dirpath='models/checkpoints',
        filename='gnn-best-{epoch:02d}-{val_roc_auc:.3f}'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[early_stop, checkpoint],
        accelerator='auto',  # Automatically uses MPS/CUDA/CPU
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=True
    )

    print(" Trainer configured")
    print(f"   Max epochs: 50")
    print(f"   Early stopping patience: 10")
    print(f"   Device: {trainer.accelerator}")

    # =========================================================================
    # STEP 6: Train!
    # =========================================================================
    print("\n STEP 6: Starting training...")
    print("   (This will take 10-30 minutes depending on your hardware)")
    print()

    trainer.fit(model, train_loader, val_loader)

    print("\n Training complete!")
    print(f"   Best model saved to: {checkpoint.best_model_path}")

    # =========================================================================
    # STEP 7: Evaluate on Test Set
    # =========================================================================
    print("\n STEP 7: Evaluating on test set...")

    # Load best model and test
    test_results = trainer.test(model, test_loader, ckpt_path='best')

    print("\n" + "=" * 70)
    print(" GNN TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Check the results above")
    print("2. Compare with baseline models (RF, XGBoost)")
    print("3. Try hyperparameter optimization:")
    print("   from Classification.src.sider_gnn_bayesian_opti import objective")
    print("4. Visualize learned embeddings with PCA/UMAP")
    print("\nSee GNN_Tutorial.md for more advanced techniques!")


if __name__ == "__main__":
    main()
