"""
Complete BACE Analysis: Baselines + GNN Comparison + Visualizations
====================================================================

This script performs comprehensive analysis of BACE pIC50 prediction:
1. Train baseline models (RF, XGBoost, SVR) with multiple featurizers
2. Compare with GNN results
3. Generate prediction analysis plots
4. Visualize learned molecular embeddings (PCA/UMAP)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from torch_geometric.loader import DataLoader

from Regression.src.baseline_regression_models import run_base_regressors
from Regression.src.bace_gnn import BACEGraphDataset, MPNN_BACE
from bace_simple_featurizer import featurize_molecules_simple

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_gnn_model(checkpoint_path, std=1.34):
    """Load trained GNN model from checkpoint"""
    # Force CPU to avoid MPS placeholder storage issue
    model = MPNN_BACE.load_from_checkpoint(checkpoint_path, std=std, map_location='cpu')
    model.eval()
    model = model.cpu()  # Ensure model is on CPU
    return model


def extract_embeddings(model, dataset):
    """Extract molecular embeddings from GNN before final MLP layer"""
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            # Ensure batch is on CPU
            batch = batch.cpu()

            # Forward pass up to graph pooling (before final MLP)
            x = model.atom_emb(batch.x)
            h = x.unsqueeze(0)
            edge_attr = model.bond_emb(batch.edge_attr)

            # Message passing
            for i in range(3):
                m = torch.relu(model.conv(x, batch.edge_index, edge_attr))
                x, h = model.gru(m.unsqueeze(0), h)
                x = x.squeeze(0)

            # Graph pooling - this is the embedding!
            from torch_geometric.nn import global_add_pool
            graph_embeddings = global_add_pool(x, batch.batch)

            embeddings.append(graph_embeddings.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    return np.vstack(embeddings), np.concatenate(labels)


def get_gnn_predictions(model, dataset, mean, std):
    """Get GNN predictions on dataset"""
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in loader:
            # Ensure batch is on CPU
            batch = batch.cpu()

            pred = model(batch)
            # Denormalize predictions
            pred_denorm = pred.cpu().numpy() * std + mean
            actual_denorm = batch.y.cpu().numpy() * std + mean

            predictions.extend(pred_denorm)
            actuals.extend(actual_denorm)

    return np.array(predictions), np.array(actuals)


def plot_baseline_comparison(baseline_results, gnn_rmse, save_path='results/baseline_comparison.png'):
    """Plot comparison of all models"""
    os.makedirs('results', exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot baseline results
    featurizers = baseline_results['Featurizer'].unique()
    models = baseline_results['Model'].unique()

    x = np.arange(len(featurizers))
    width = 0.25

    for i, model_name in enumerate(models):
        model_data = baseline_results[baseline_results['Model'] == model_name]
        rmse_values = [model_data[model_data['Featurizer'] == f]['RMSE'].values[0]
                       for f in featurizers]
        ax.bar(x + i*width, rmse_values, width, label=model_name, alpha=0.8)

    # Add GNN result as horizontal line
    ax.axhline(y=gnn_rmse, color='red', linestyle='--', linewidth=2, label='GNN (MPNN)')

    ax.set_xlabel('Featurizer', fontweight='bold', fontsize=12)
    ax.set_ylabel('RMSE (pIC50 units)', fontweight='bold', fontsize=12)
    ax.set_title('BACE pIC50 Prediction: Baseline Models vs GNN', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(featurizers, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved baseline comparison plot to: {save_path}")
    plt.close()


def plot_predictions(predictions, actuals, title, save_path):
    """Plot predicted vs actual values"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot
    ax.scatter(actuals, predictions, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    # Perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # Add metrics text
    textstr = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    ax.set_xlabel('Actual pIC50', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted pIC50', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved prediction plot to: {save_path}")
    plt.close()


def plot_residuals(predictions, actuals, save_path):
    """Plot residual distribution"""
    residuals = actuals - predictions

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Residual scatter
    ax1.scatter(actuals, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Actual pIC50', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Residual (Actual - Predicted)', fontweight='bold', fontsize=12)
    ax1.set_title('Residual Plot', fontweight='bold', fontsize=14)
    ax1.grid(alpha=0.3)

    # Residual histogram
    ax2.hist(residuals, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax2.set_title('Residual Distribution', fontweight='bold', fontsize=14)
    ax2.grid(alpha=0.3)

    # Add stats
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    textstr = f'Mean = {mean_res:.3f}\nStd = {std_res:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.70, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved residual plot to: {save_path}")
    plt.close()


def plot_embeddings(embeddings, labels, method='PCA', save_path=None):
    """Visualize molecular embeddings"""
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
        title = 'PCA of GNN Molecular Embeddings'
    elif method == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        title = 't-SNE of GNN Molecular Embeddings'
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        title = 'UMAP of GNN Molecular Embeddings'
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"   Running {method} dimensionality reduction...")
    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color by pIC50 value
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=labels, cmap='viridis', s=50, alpha=0.7,
                        edgecolors='k', linewidth=0.5)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('pIC50', fontweight='bold', fontsize=12)

    ax.set_xlabel(f'{method} Component 1', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'{method} Component 2', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)

    # Add variance explained for PCA
    if method == 'PCA':
        var_exp = reducer.explained_variance_ratio_
        textstr = f'Variance explained:\nPC1: {var_exp[0]:.2%}\nPC2: {var_exp[1]:.2%}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {method} embedding plot to: {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("BACE Complete Analysis: Baselines + GNN + Visualizations")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\nSTEP 1: Loading BACE data...")
    df = pd.read_csv('data/raw/bace.csv')
    print(f"Loaded {len(df)} molecules")

    # =========================================================================
    # STEP 2: Train Baseline Models with Multiple Featurizers
    # =========================================================================
    print("\nSTEP 2: Training baseline models with multiple featurizers...")
    print("   This will take 5-10 minutes...")

    # Generate features
    print("\n   Generating molecular features...")
    # Rename column temporarily for featurizer
    df_temp = df.rename(columns={'mol': 'canonical_smiles'})
    features_dict = featurize_molecules_simple(
        df_temp,
        smiles_col='canonical_smiles',
        target_col='pIC50',
        methods=['ecfp', 'maccs', 'rdkit']
    )

    print(f"   Generated features for {len(features_dict)} featurizers")
    for feat_name, feat_df in features_dict.items():
        print(f"     - {feat_name}: {feat_df.shape[1]-2} features")

    # Train baseline models
    print("\n   Training baseline models (RF, XGBoost, SVR)...")
    baseline_results, trained_models = run_base_regressors(features_dict, verbose=True)

    # =========================================================================
    # STEP 3: Load GNN Results
    # =========================================================================
    print("\nSTEP 3: Loading GNN model and results...")

    # Find best checkpoint
    import glob
    checkpoints = glob.glob('models/checkpoints/bace-gnn-best-*.ckpt')
    if not checkpoints:
        print("Error: No GNN checkpoint found! Please train the GNN first.")
        return

    checkpoint_path = sorted(checkpoints)[-1]  # Get most recent
    print(f"   Using checkpoint: {checkpoint_path}")

    # Load dataset
    dataset = BACEGraphDataset(
        root='data/processed/bace_graphs',
        df=df,
        smiles_col='mol',
        target_col='pIC50'
    )

    # Calculate normalization stats
    mean = df['pIC50'].mean()
    std = df['pIC50'].std()

    # Load model
    model = load_gnn_model(checkpoint_path, std=std)
    print(f"Loaded GNN model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Get GNN predictions
    print("   Generating GNN predictions...")
    gnn_predictions, gnn_actuals = get_gnn_predictions(model, dataset, mean, std)
    gnn_rmse = np.sqrt(np.mean((gnn_predictions - gnn_actuals)**2))
    print(f"   GNN RMSE: {gnn_rmse:.4f}")

    # =========================================================================
    # STEP 4: Generate Comparison Plots
    # =========================================================================
    print("\nSTEP 4: Generating comparison plots...")

    # Baseline comparison
    plot_baseline_comparison(baseline_results, gnn_rmse, 'results/baseline_comparison.png')

    # GNN predictions
    plot_predictions(
        gnn_predictions, gnn_actuals,
        'GNN (MPNN) Predictions vs Actual pIC50',
        'results/gnn_predictions.png'
    )

    # Residuals
    plot_residuals(gnn_predictions, gnn_actuals, 'results/gnn_residuals.png')

    # Best baseline predictions
    best_baseline = baseline_results.iloc[0]
    print(f"\n   Best baseline: {best_baseline['Model']} with {best_baseline['Featurizer']}")
    print(f"   Best baseline RMSE: {best_baseline['RMSE']:.4f}")

    # =========================================================================
    # STEP 5: Extract and Visualize Embeddings
    # =========================================================================
    print("\nSTEP 5: Extracting and visualizing GNN embeddings...")

    print("   Extracting embeddings from GNN...")
    embeddings, labels = extract_embeddings(model, dataset)
    print(f"   Extracted embeddings: {embeddings.shape}")

    # Denormalize labels for visualization
    labels_denorm = labels * std + mean

    # Generate embedding visualizations
    plot_embeddings(embeddings, labels_denorm, method='PCA',
                   save_path='results/embeddings_pca.png')
    plot_embeddings(embeddings, labels_denorm, method='UMAP',
                   save_path='results/embeddings_umap.png')

    # =========================================================================
    # STEP 6: Summary Report
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)

    print("\nModel Performance Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'Featurizer':<15} {'RMSE':>10} {'Improvement vs Best Baseline':>25}")
    print("-" * 80)

    # Show top 5 baselines
    for idx, row in baseline_results.head(5).iterrows():
        print(f"{row['Model']:<20} {row['Featurizer']:<15} {row['RMSE']:>10.4f} {'-':>25}")

    best_baseline_rmse = baseline_results['RMSE'].min()
    improvement = ((best_baseline_rmse - gnn_rmse) / best_baseline_rmse) * 100

    print("-" * 80)
    print(f"{'GNN (MPNN)':<20} {'Graph':<15} {gnn_rmse:>10.4f} {improvement:>24.1f}%")
    print("=" * 80)

    print("\nGenerated Files:")
    print("   - results/baseline_comparison.png - Model comparison")
    print("   - results/gnn_predictions.png - GNN predictions vs actual")
    print("   - results/gnn_residuals.png - Residual analysis")
    print("   - results/embeddings_pca.png - PCA of molecular embeddings")
    print("   - results/embeddings_umap.png - UMAP of molecular embeddings")

    print("\nComplete analysis finished!")
    print("=" * 80)


if __name__ == "__main__":
    main()
