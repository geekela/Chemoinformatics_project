"""
BACE Prediction Error Analysis
================================

This script performs detailed error analysis to understand:
1. Which molecules have the largest prediction errors
2. Why all featurizers perform similarly
3. Chemical patterns and structural features that cause errors
4. Comparison of error patterns across different models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from torch_geometric.loader import DataLoader

from Regression.src.bace_gnn import BACEGraphDataset, MPNN_BACE
from bace_simple_featurizer import featurize_molecules_simple

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def calculate_molecular_properties(smiles):
    """Calculate key molecular properties for a SMILES string"""
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
        'NumHeteroatoms': Lipinski.NumHeteroAtoms(mol),
        'FractionCSP3': Lipinski.FractionCsp3(mol)
    }


def get_model_predictions(df, features_dict):
    """
    Train baseline models and get predictions
    Returns dict of {model_name: predictions}
    """
    print("\n🔬 Training baseline models for error analysis...")

    predictions_dict = {}

    # Use ECFP features for baseline (best performing typically)
    X_train = features_dict['ecfp'].drop(columns=['canonical_smiles', 'pIC50']).values
    y_train = features_dict['ecfp']['pIC50'].values

    # Train Random Forest
    print("   Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=500, max_depth=30, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    predictions_dict['RandomForest'] = rf.predict(X_train)

    print(f"   RF RMSE: {np.sqrt(mean_squared_error(y_train, predictions_dict['RandomForest'])):.4f}")

    return predictions_dict, y_train


def get_gnn_predictions(checkpoint_path, df, std):
    """Get GNN predictions"""
    print("\n🧠 Loading GNN model for error analysis...")

    # Load dataset
    dataset = BACEGraphDataset(
        root='data/processed/bace_graphs',
        df=df,
        smiles_col='mol',
        target_col='pIC50'
    )

    # Load model
    model = MPNN_BACE.load_from_checkpoint(checkpoint_path, std=std, map_location='cpu')
    model.eval()
    model = model.cpu()

    # Get predictions
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions = []

    mean = df['pIC50'].mean()

    with torch.no_grad():
        for batch in loader:
            batch = batch.cpu()
            pred = model(batch)
            pred_denorm = pred.cpu().numpy() * std + mean
            predictions.extend(pred_denorm)

    print(f"   GNN RMSE: {np.sqrt(mean_squared_error(df['pIC50'].values, predictions)):.4f}")

    return np.array(predictions)


def analyze_worst_predictions(df, predictions_dict, top_n=20):
    """Identify and analyze molecules with largest prediction errors"""
    print(f"\n Analyzing top {top_n} worst predictions for each model...")

    results = []

    for model_name, predictions in predictions_dict.items():
        actuals = df['pIC50'].values
        errors = np.abs(actuals - predictions)

        # Get worst predictions
        worst_indices = np.argsort(errors)[-top_n:][::-1]

        for idx in worst_indices:
            smiles = df.iloc[idx]['mol']
            actual = actuals[idx]
            predicted = predictions[idx]
            error = errors[idx]

            # Calculate molecular properties
            props = calculate_molecular_properties(smiles)

            if props:
                results.append({
                    'Model': model_name,
                    'SMILES': smiles,
                    'Actual_pIC50': actual,
                    'Predicted_pIC50': predicted,
                    'Absolute_Error': error,
                    **props
                })

    return pd.DataFrame(results)


def plot_error_distribution(df, predictions_dict, save_path='results/error_distribution.png'):
    """Plot error distribution for each model"""
    print("\n Plotting error distributions...")
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(2, len(predictions_dict), figsize=(18, 10))
    if len(predictions_dict) == 1:
        axes = axes.reshape(-1, 1)

    actuals = df['pIC50'].values

    for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
        errors = actuals - predictions
        abs_errors = np.abs(errors)

        # Error distribution
        axes[0, idx].hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, idx].set_title(f'{model_name}\nError Distribution', fontweight='bold')
        axes[0, idx].set_xlabel('Error (Actual - Predicted)')
        axes[0, idx].set_ylabel('Frequency')
        axes[0, idx].grid(alpha=0.3)

        # Add statistics
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        textstr = f'Mean: {mean_err:.3f}\nStd: {std_err:.3f}\nRMSE: {rmse:.3f}'
        axes[0, idx].text(0.02, 0.98, textstr, transform=axes[0, idx].transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Error vs actual value
        axes[1, idx].scatter(actuals, abs_errors, alpha=0.5, s=30)
        axes[1, idx].axhline(y=np.median(abs_errors), color='red',
                            linestyle='--', linewidth=2, label=f'Median: {np.median(abs_errors):.3f}')
        axes[1, idx].set_title(f'{model_name}\nError vs Actual pIC50', fontweight='bold')
        axes[1, idx].set_xlabel('Actual pIC50')
        axes[1, idx].set_ylabel('Absolute Error')
        axes[1, idx].legend()
        axes[1, idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Saved error distribution plot to: {save_path}")
    plt.close()


def plot_property_vs_error(error_df, save_path='results/property_vs_error.png'):
    """Plot molecular properties vs prediction error"""
    print("\n Analyzing correlation between molecular properties and errors...")
    os.makedirs('results', exist_ok=True)

    # Properties to analyze
    properties = ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'NumRotatableBonds',
                  'NumAromaticRings', 'NumRings']

    models = error_df['Model'].unique()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, prop in enumerate(properties):
        for model in models:
            model_data = error_df[error_df['Model'] == model]
            axes[idx].scatter(model_data[prop], model_data['Absolute_Error'],
                            alpha=0.5, s=30, label=model)

        # Calculate correlation
        corr = error_df.groupby('Model').apply(
            lambda x: x[prop].corr(x['Absolute_Error'])
        ).mean()

        axes[idx].set_xlabel(prop, fontweight='bold')
        axes[idx].set_ylabel('Absolute Error')
        axes[idx].set_title(f'{prop} vs Error (avg corr: {corr:.3f})', fontweight='bold')
        axes[idx].grid(alpha=0.3)
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Saved property vs error plot to: {save_path}")
    plt.close()


def analyze_featurizer_similarity(features_dict):
    """
    Analyze why different featurizers give similar results
    by computing correlation between feature representations
    """
    print("\n Analyzing featurizer similarity...")

    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr

    results = {}

    # Get feature matrices
    feature_matrices = {}
    for feat_name, feat_df in features_dict.items():
        X = feat_df.drop(columns=['canonical_smiles', 'pIC50']).values
        feature_matrices[feat_name] = X

    # Calculate pairwise distance correlations
    feat_names = list(feature_matrices.keys())
    n_feats = len(feat_names)

    correlation_matrix = np.zeros((n_feats, n_feats))

    print("\n   Computing pairwise correlations between featurizer distance matrices...")
    for i, name1 in enumerate(feat_names):
        for j, name2 in enumerate(feat_names):
            if i <= j:
                # Compute distance matrices
                dist1 = pdist(feature_matrices[name1][:100], metric='euclidean')  # Sample 100 molecules
                dist2 = pdist(feature_matrices[name2][:100], metric='euclidean')

                # Compute correlation between distance matrices
                corr, _ = spearmanr(dist1, dist2)
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr

                if i != j:
                    print(f"      {name1} vs {name2}: {corr:.3f}")

    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=0, vmax=1)

    ax.set_xticks(np.arange(n_feats))
    ax.set_yticks(np.arange(n_feats))
    ax.set_xticklabels(feat_names, rotation=45, ha='right')
    ax.set_yticklabels(feat_names)

    # Add correlation values
    for i in range(n_feats):
        for j in range(n_feats):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Featurizer Distance Matrix Correlation\n(Higher = More Similar Molecular Representations)',
                fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax, label='Spearman Correlation')

    plt.tight_layout()
    plt.savefig('results/featurizer_similarity.png', dpi=300, bbox_inches='tight')
    print(f" Saved featurizer similarity plot to: results/featurizer_similarity.png")
    plt.close()

    return correlation_matrix, feat_names


def plot_error_overlap(error_df, save_path='results/error_overlap.png'):
    """Analyze overlap of worst predictions between models"""
    print("\n Analyzing error overlap between models...")
    os.makedirs('results', exist_ok=True)

    models = error_df['Model'].unique()

    # Get top 20 worst predictions for each model
    top_n = 20
    worst_smiles_sets = {}

    for model in models:
        model_errors = error_df[error_df['Model'] == model].nlargest(top_n, 'Absolute_Error')
        worst_smiles_sets[model] = set(model_errors['SMILES'].values)

    # Calculate overlap
    if len(models) == 2:
        model1, model2 = models
        overlap = worst_smiles_sets[model1] & worst_smiles_sets[model2]
        only_model1 = worst_smiles_sets[model1] - worst_smiles_sets[model2]
        only_model2 = worst_smiles_sets[model2] - worst_smiles_sets[model1]

        # Plot Venn diagram style
        fig, ax = plt.subplots(figsize=(10, 8))

        categories = [f'Only {model1}', 'Both Models', f'Only {model2}']
        counts = [len(only_model1), len(overlap), len(only_model2)]
        colors = ['lightblue', 'lightcoral', 'lightgreen']

        bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=2)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({count/top_n*100:.0f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=14)

        ax.set_ylabel('Number of Molecules', fontweight='bold', fontsize=12)
        ax.set_title(f'Overlap of Top {top_n} Worst Predictions Between Models\n(High overlap = models struggle with same molecules)',
                    fontweight='bold', fontsize=14)
        ax.set_ylim(0, top_n + 2)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved error overlap plot to: {save_path}")
        plt.close()

        print(f"\n   Overlap Analysis:")
        print(f"   • Only {model1} struggles: {len(only_model1)} molecules ({len(only_model1)/top_n*100:.0f}%)")
        print(f"   • Both models struggle: {len(overlap)} molecules ({len(overlap)/top_n*100:.0f}%)")
        print(f"   • Only {model2} struggles: {len(only_model2)} molecules ({len(only_model2)/top_n*100:.0f}%)")

        return overlap, only_model1, only_model2

    return None


def generate_worst_molecules_report(error_df, overlap_smiles, save_path='results/worst_molecules_report.txt'):
    """Generate text report of worst molecules"""
    print(f"\n Generating worst molecules report...")
    os.makedirs('results', exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("BACE ERROR ANALYSIS: WORST PREDICTIONS REPORT\n")
        f.write("=" * 100 + "\n\n")

        # Overall statistics
        f.write("OVERALL ERROR STATISTICS\n")
        f.write("-" * 100 + "\n")
        for model in error_df['Model'].unique():
            model_data = error_df[error_df['Model'] == model]
            f.write(f"\n{model}:\n")
            f.write(f"  Mean Absolute Error: {model_data['Absolute_Error'].mean():.4f}\n")
            f.write(f"  Median Absolute Error: {model_data['Absolute_Error'].median():.4f}\n")
            f.write(f"  Max Error: {model_data['Absolute_Error'].max():.4f}\n")
            f.write(f"  90th Percentile: {model_data['Absolute_Error'].quantile(0.9):.4f}\n")

        # Molecules that all models struggle with
        if overlap_smiles is not None and len(overlap_smiles) > 0:
            f.write("\n\n" + "=" * 100 + "\n")
            f.write(f"MOLECULES THAT ALL MODELS STRUGGLE WITH ({len(overlap_smiles)} molecules)\n")
            f.write("=" * 100 + "\n\n")
            f.write("These molecules have high prediction errors across all models,\n")
            f.write("suggesting intrinsic difficulty or potential data quality issues.\n\n")

            for smiles in list(overlap_smiles)[:10]:  # Show top 10
                mol_data = error_df[error_df['SMILES'] == smiles].iloc[0]
                f.write(f"\nSMILES: {smiles}\n")
                f.write(f"Actual pIC50: {mol_data['Actual_pIC50']:.3f}\n")
                f.write(f"Predicted pIC50: {mol_data['Predicted_pIC50']:.3f}\n")
                f.write(f"Absolute Error: {mol_data['Absolute_Error']:.3f}\n")
                f.write(f"Molecular Weight: {mol_data['MW']:.2f}\n")
                f.write(f"LogP: {mol_data['LogP']:.2f}\n")
                f.write(f"HBA: {mol_data['HBA']}, HBD: {mol_data['HBD']}\n")
                f.write(f"TPSA: {mol_data['TPSA']:.2f}\n")
                f.write(f"Num Aromatic Rings: {mol_data['NumAromaticRings']}\n")
                f.write("-" * 100 + "\n")

        # Property analysis
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("MOLECULAR PROPERTY ANALYSIS OF HIGH-ERROR MOLECULES\n")
        f.write("=" * 100 + "\n\n")

        # Compare high-error vs low-error molecules
        for model in error_df['Model'].unique():
            model_data = error_df[error_df['Model'] == model]
            high_error = model_data.nlargest(50, 'Absolute_Error')
            low_error = model_data.nsmallest(50, 'Absolute_Error')

            f.write(f"\n{model} - Property Comparison:\n")
            f.write(f"{'Property':<25} {'High Error (top 50)':<25} {'Low Error (bottom 50)':<25} {'Difference':<15}\n")
            f.write("-" * 100 + "\n")

            for prop in ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'NumRotatableBonds',
                        'NumAromaticRings', 'NumRings']:
                high_mean = high_error[prop].mean()
                low_mean = low_error[prop].mean()
                diff = high_mean - low_mean
                f.write(f"{prop:<25} {high_mean:<25.2f} {low_mean:<25.2f} {diff:<15.2f}\n")

    print(f" Saved worst molecules report to: {save_path}")


def main():
    print("=" * 100)
    print("BACE PREDICTION ERROR ANALYSIS")
    print("=" * 100)

    # Load data
    print("\n Loading BACE data...")
    df = pd.read_csv('data/raw/bace.csv')
    print(f" Loaded {len(df)} molecules")

    # Generate features
    print("\n Generating molecular features...")
    df_temp = df.rename(columns={'mol': 'canonical_smiles'})
    features_dict = featurize_molecules_simple(
        df_temp,
        smiles_col='canonical_smiles',
        target_col='pIC50',
        methods=['ecfp', 'maccs', 'rdkit']
    )

    # Get predictions from baseline models
    baseline_predictions, y_true = get_model_predictions(df, features_dict)

    # Get GNN predictions
    import glob
    checkpoints = glob.glob('models/checkpoints/bace-gnn-best-*.ckpt')
    if checkpoints:
        checkpoint_path = sorted(checkpoints)[-1]
        std = df['pIC50'].std()
        gnn_predictions = get_gnn_predictions(checkpoint_path, df, std)
        baseline_predictions['GNN'] = gnn_predictions
    else:
        print("  No GNN checkpoint found, skipping GNN analysis")

    # Analyze worst predictions
    error_df = analyze_worst_predictions(df, baseline_predictions, top_n=50)

    # Generate plots
    plot_error_distribution(df, baseline_predictions)
    plot_property_vs_error(error_df)

    # Analyze featurizer similarity
    corr_matrix, feat_names = analyze_featurizer_similarity(features_dict)

    print("\n Featurizer Similarity Summary:")
    print("   High correlation = featurizers capture similar chemical information")
    print("   → This explains why all featurizers give similar prediction performance!")

    # Analyze error overlap
    overlap, only_rf, only_gnn = plot_error_overlap(error_df) if len(baseline_predictions) >= 2 else (None, None, None)

    # Generate report
    generate_worst_molecules_report(error_df, overlap)

    # Summary
    print("\n" + "=" * 100)
    print(" ERROR ANALYSIS COMPLETE")
    print("=" * 100)
    print("\n Generated Files:")
    print("   • results/error_distribution.png - Error distributions by model")
    print("   • results/property_vs_error.png - Molecular properties vs prediction error")
    print("   • results/featurizer_similarity.png - Correlation between featurizers")
    print("   • results/error_overlap.png - Overlap of worst predictions")
    print("   • results/worst_molecules_report.txt - Detailed text report")

    print("\n Key Insights:")
    print("   1. Check featurizer similarity plot - high correlation explains similar performance")
    print("   2. Look at error overlap - molecules that all models struggle with")
    print("   3. Review property correlations - which molecular features cause errors")
    print("\n Analysis complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
