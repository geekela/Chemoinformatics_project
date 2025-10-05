"""
BACE Dataset Preprocessing Pipeline
Date: October 5, 2025

OBJECTIVE: Clean and preprocess the BACE dataset for a regression task (pIC50) 
by generating basic molecular properties (Featurization 1).
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 

from rdkit import Chem
from rdkit.Chem import Descriptors # Needed for basic molecular properties
from rdkit.Chem import AllChem # Kept as it is often needed for RDKit operations

# ----------------------------------------------------------------------
# Suppress RDKit deprecation warning for cleaner logs
# ----------------------------------------------------------------------
warnings.filterwarnings(
    "ignore", 
    category=DeprecationWarning, 
    message="^DEPRECATION WARNING: please use MorganGenerator"
)
# ----------------------------------------------------------------------


class BACEPreprocessor:
    def __init__(self, input_path, output_dir, target_col='pIC50'):
        """
        Initializes the preprocessor.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.target_col = target_col
        self.report = []
        
        self.df = None
        self.df_f1 = None 
        
    def log(self, message):
        """Logs the preprocessing steps."""
        date_prefix = "2025-10-05" 
        timestamp = datetime.now().strftime(f"{date_prefix} %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.report.append(log_message)
    
    def load_data(self):
        """Loads the BACE dataset."""
        self.log("Loading BACE dataset...")
        try:
            self.df = pd.read_csv(self.input_path)
            
            if 'mol' in self.df.columns:
                self.df = self.df.rename(columns={'mol': 'smiles'})
            
            if 'smiles' not in self.df.columns:
                raise ValueError(f"SMILES column not found in dataset.")
                
            if self.target_col not in self.df.columns:
                raise ValueError(f"Target column '{self.target_col}' not found.")
                
            self.log(f"Loaded {len(self.df)} molecules.")
            return self.df
        except Exception as e:
            self.log(f"Error loading data: {e}")
            raise
    
    def validate_smiles(self):
        """Validates SMILES and removes invalid entries."""
        self.log("Validating SMILES...")
        valid_indices = []
        
        for idx, smiles in enumerate(self.df['smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_indices.append(idx)
        
        invalid_count = len(self.df) - len(valid_indices)
        self.log(f"Found {invalid_count} invalid SMILES, removing them.")
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        return self.df
    
    def canonicalize_and_deduplicate(self):
        """Canonicalizes SMILES strings and removes duplicates."""
        self.log("Canonicalizing SMILES...")
        
        canonical_smiles = []
        for smiles in self.df['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            else:
                canonical_smiles.append(None) 
        
        self.df['canonical_smiles'] = canonical_smiles
        self.df = self.df.dropna(subset=['canonical_smiles']) 
        
        original_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['canonical_smiles'], keep='first')
        removed_count = original_count - len(self.df)
        
        self.log(f"Removed {removed_count} duplicate entries based on canonical SMILES.")
        return self.df.reset_index(drop=True)
    
    def analyze_target_distribution(self):
        """Analyzes the continuous target distribution (pIC50)."""
        self.log(f"Analyzing target distribution for {self.target_col}...")
        
        self.df = self.df.dropna(subset=[self.target_col]).reset_index(drop=True)
        
        self.target_stats = self.df[self.target_col].describe().to_frame().T
        self.log(f"Target statistics:\n{self.target_stats.to_string()}")
        
        return self.target_stats

    # ----------------------------------------------------------------------
    # FEATURISATION 1 (F1 - Basic Properties)
    # ----------------------------------------------------------------------
    def add_molecular_properties_f1(self):
        """Calculates RDKit molecular descriptors."""
        self.log("Calculating basic molecular properties...")
        
        mols = [Chem.MolFromSmiles(s) for s in self.df['canonical_smiles']]
        
        # Get all RDKit descriptors
        descriptor_names = [d[0] for d in Descriptors.descList]
        descriptor_funcs = [d[1] for d in Descriptors.descList]
        
        data = {}
        num_molecules = len(mols)
        
        for name, func in zip(descriptor_names, descriptor_funcs):
            try:
                data[name] = [func(m) for m in mols]
            except Exception:
                self.log(f"Warning: Could not compute descriptor {name}. Filling with NaN.")
                data[name] = [np.nan] * num_molecules 

        df_temp = self.df[['canonical_smiles', self.target_col]].copy()
        df_descriptors = pd.DataFrame(data)
        
        # Concatenate descriptors
        self.df_f1 = pd.concat([df_temp.reset_index(drop=True), df_descriptors.reset_index(drop=True)], axis=1)
        
        # Remove columns that are entirely NaN
        self.df_f1 = self.df_f1.dropna(axis=1, how='all')
        
        self.log(f"Descriptors added. Processed DataFrame shape: {self.df_f1.shape}")
        return self.df_f1

    # ----------------------------------------------------------------------
    # SAVING AND VISUALIZATION
    # ----------------------------------------------------------------------
    def create_visualizations(self):
        """Creates summary visualizations."""
        self.log("Creating visualizations...")
        if self.df_f1 is None:
            self.log("WARNING: Data not computed, skipping visualizations.")
            return

        df_plot = self.df_f1.copy()
        
        # Select key properties for visualization
        plot_cols = [self.target_col, 'MolWt', 'HeavyAtomCount']
        df_plot = df_plot[[c for c in plot_cols if c in df_plot.columns]]
        
        # Plot target distribution and key features
        if len(df_plot.columns) < 3:
            self.log("Not enough columns for detailed visualization, plotting only target.")
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
            sns.histplot(self.df_f1[self.target_col], bins=30, kde=True, ax=ax1, color='skyblue', edgecolor='black')
            ax1.set_xlabel(self.target_col)
            ax1.set_title(f'Distribution of Target Variable ({self.target_col})')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'bace_preprocessing_summary.png'), dpi=300)
            plt.close()
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Target Distribution
        ax1 = axes[0, 0]
        sns.histplot(df_plot[self.target_col], bins=30, kde=True, ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_xlabel(self.target_col)
        ax1.set_title(f'Distribution of Target Variable ({self.target_col})')
        
        # 2. Molecular Weight
        ax2 = axes[0, 1]
        sns.histplot(df_plot['MolWt'], bins=50, kde=True, ax=ax2, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Molecular Weight (MolWt)')
        ax2.set_title('Molecular Weight Distribution')
        
        # 3. Heavy Atom Count
        ax3 = axes[1, 0]
        sns.histplot(df_plot['HeavyAtomCount'], bins=30, kde=False, ax=ax3, color='mediumseagreen', edgecolor='black')
        ax3.set_xlabel('Number of Heavy Atoms (HeavyAtomCount)')
        ax3.set_title('Heavy Atom Count Distribution')
        
        # 4. MW vs. pIC50
        ax4 = axes[1, 1]
        sns.scatterplot(x='MolWt', y=self.target_col, data=df_plot, ax=ax4, alpha=0.6, color='darkblue')
        ax4.set_xlabel('Molecular Weight (MolWt)')
        ax4.set_ylabel(self.target_col)
        ax4.set_title(f'MW vs. {self.target_col}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'bace_preprocessing_summary.png'), dpi=300)
        plt.close()
    
    def save_results(self):
        """Saves the processed data and reports."""
        self.log("Saving results...")
        
        # Save processed data (F1)
        if self.df_f1 is not None:
            output_path = os.path.join(self.output_dir, 'bace_processed_data.csv')
            df_to_save = self.df_f1.copy()
            df_to_save.to_csv(output_path, index=False)
            self.log(f"Saved processed data to {output_path} (Shape: {df_to_save.shape})")
        else:
            self.log("WARNING: Processed data is missing. Nothing to save.")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'bace_preprocessing_report.txt')
        with open(report_path, 'w') as f:
            f.write("BACE Dataset Preprocessing Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(self.report))
            f.write("\n\nTarget Distribution Summary:\n")
            if hasattr(self, 'target_stats'):
                f.write(self.target_stats.to_string())
            else:
                f.write("Statistics not computed.")
        
        # Save target statistics
        stats_path = os.path.join(self.output_dir, 'bace_target_stats.csv')
        if hasattr(self, 'target_stats'):
            self.target_stats.to_csv(stats_path)
    
    def run_pipeline(self):
        """Runs the complete preprocessing pipeline."""
        self.log("--- BACE PREPROCESSING PIPELINE STARTING ---")
        
        self.load_data()
        self.validate_smiles()
        self.canonicalize_and_deduplicate()
        self.analyze_target_distribution()
        
        self.add_molecular_properties_f1()
        
        self.create_visualizations()
        self.save_results()
        
        self.log("--- BACE PREPROCESSING PIPELINE COMPLETE ---")
        return self.df_f1

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True) 
    
    # --- Create a mock bace.csv file for testing if missing ---
    if not os.path.exists('data/raw/bace.csv'):
        print("\nNOTE: Creating a mock bace.csv file for execution...")
        mock_data = {
            'mol': ['COc1cc(OC)c(cc1C(=O)NCCCNC(=O)c1ccccc1)C', 'CC(=O)NCC1(CCN(C)CC1)c1ccc(C)cc1'],
            'pIC50': [7.5, 6.2],
            'Class': [1, 1],
            'Standard Value': [31.62, 630.95]
        }
        mock_df = pd.DataFrame(mock_data)
        mock_df.to_csv('data/raw/bace.csv', index=False)
    # ------------------------------------------------------------------------

    try:
        preprocessor = BACEPreprocessor(
            input_path='data/raw/bace.csv',
            output_dir='data/processed/'
        )
        preprocessor.run_pipeline() 
        
    except FileNotFoundError:
        print("\nERROR: The file 'data/raw/bace.csv' is missing. Please place it there before running.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
