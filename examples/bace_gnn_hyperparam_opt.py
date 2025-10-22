"""
GNN Hyperparameter Optimization for BACE using Optuna
=======================================================

This script performs Bayesian optimization to find the best hyperparameters
for the BACE GNN model.

Hyperparameters to optimize:
- hidden_dim: Embedding dimension (32, 64, 128, 256)
- learning_rate: Learning rate (1e-4 to 1e-2)
- batch_size: Batch size (16, 32, 64)
- num_layers: Number of message passing layers (2, 3, 4, 5)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from Regression.src.bace_gnn import BACEGraphDataset, MPNN_BACE


def objective(trial):
    """
    Optuna objective function - trains a GNN with sampled hyperparameters
    and returns validation RMSE
    """
    # Sample hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    print(f"\nTrial {trial.number}: hidden_dim={hidden_dim}, lr={lr:.5f}, batch_size={batch_size}")

    # Load data (use cached graphs)
    df = pd.read_csv('data/raw/bace.csv')

    dataset = BACEGraphDataset(
        root='data/processed/bace_graphs',
        df=df,
        smiles_col='mol',
        target_col='pIC50'
    )

    # Normalize targets
    mean = dataset.data.y.mean().item()
    std = dataset.data.y.std().item()
    dataset.data.y = (dataset.data.y - mean) / std

    # Split
    dataset = dataset.shuffle()
    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    # Create loaders
    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset[val_idx], batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = MPNN_BACE(hidden_dim=hidden_dim, out_dim=1, std=std, lr=lr)

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_rmse',
        patience=10,
        mode='min',
        verbose=False
    )

    # Trainer (reduced epochs for faster optimization)
    trainer = pl.Trainer(
        max_epochs=30,  # Reduced for faster search
        callbacks=[early_stop],
        accelerator='auto',
        devices=1,
        enable_progress_bar=False,  # Disable for cleaner output
        enable_model_summary=False,
        logger=False  # Disable logging for trials
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Get best validation RMSE
    best_val_rmse = trainer.callback_metrics.get('val_rmse', float('inf'))

    print(f"   Best val_rmse: {best_val_rmse:.4f}")

    return best_val_rmse.item()


def main():
    print("=" * 80)
    print("BACE GNN Hyperparameter Optimization using Bayesian Search (Optuna)")
    print("=" * 80)

    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',  # Minimize RMSE
        study_name='bace_gnn_optimization'
    )

    # Run optimization
    print("\n🔍 Starting hyperparameter search...")
    print("   This will run 20 trials (approx. 30-60 minutes)")
    print()

    n_trials = 20
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 80)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("=" * 80)

    print("\n🏆 Best Trial:")
    print(f"   Trial number: {study.best_trial.number}")
    print(f"   Best RMSE: {study.best_value:.4f}")
    print("\n   Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"     • {key}: {value}")

    # Show top 5 trials
    print("\n📊 Top 5 Trials:")
    print("-" * 80)
    print(f"{'Trial':<8} {'RMSE':<10} {'hidden_dim':<12} {'lr':<12} {'batch_size':<12}")
    print("-" * 80)

    trials_sorted = sorted(study.trials, key=lambda t: t.value)[:5]
    for trial in trials_sorted:
        print(f"{trial.number:<8} {trial.value:<10.4f} "
              f"{trial.params.get('hidden_dim', 'N/A'):<12} "
              f"{trial.params.get('learning_rate', 0):<12.6f} "
              f"{trial.params.get('batch_size', 'N/A'):<12}")

    print("=" * 80)

    # Save study
    import joblib
    os.makedirs('results', exist_ok=True)
    joblib.dump(study, 'results/bace_gnn_study.pkl')
    print("\n✅ Study saved to: results/bace_gnn_study.pkl")

    # Generate visualizations
    print("\n📈 Generating optimization visualizations...")

    # Optimization history
    fig = plot_optimization_history(study)
    fig.write_image('results/optimization_history.png')
    print("   • Saved: results/optimization_history.png")

    # Parameter importances
    try:
        fig = plot_param_importances(study)
        fig.write_image('results/param_importances.png')
        print("   • Saved: results/param_importances.png")
    except:
        print("   • Parameter importances requires more trials (skipped)")

    print("\n💡 Next Steps:")
    print("   1. Train final model with best hyperparameters")
    print("   2. Evaluate on test set")
    print("   3. Compare with baseline GNN (hidden_dim=64, lr=1e-3)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
