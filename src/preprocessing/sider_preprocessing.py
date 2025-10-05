"""
SIDER Dataset Preprocessing Pipeline
Author: Elisa 
Date: October 5, 2025
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class SIDERPreprocessor:
    def __init__(self, input_path, output_dir):
        """
        Initialize the SIDER preprocessor
        
        Args:
            input_path (str): Path to the raw SIDER dataset
            output_dir (str): Directory to save processed data and reports
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.report = []
        
    def log(self, message):
        """
        Log preprocessing steps with timestamp
        
        Args:
            message (str): Log message to record
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.report.append(log_message)
    
    def load_data(self):
        """
        Load the raw SIDER dataset from CSV file
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        self.log("Loading SIDER dataset...")
        self.df = pd.read_csv(self.input_path)
        self.log(f"Loaded {len(self.df)} molecules with {self.df.shape[1]-1} ADR classes")
        return self.df
    
    def validate_smiles(self):
        """
        Validate SMILES strings and remove invalid molecules
        
        Returns:
            pd.DataFrame: Dataset with valid SMILES only
        """
        self.log("Validating SMILES...")
        valid_indices = []
        invalid_smiles = []
        
        for idx, smiles in enumerate(self.df['smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_indices.append(idx)
            else:
                invalid_smiles.append((idx, smiles))
        
        self.log(f"Found {len(invalid_smiles)} invalid SMILES")
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        self.invalid_smiles = invalid_smiles
        return self.df
    
    def canonicalize_and_deduplicate(self):
        """
        Canonicalize SMILES strings and remove duplicate molecules
        
        Returns:
            pd.DataFrame: Dataset with canonical SMILES and duplicates removed
        """
        self.log("Canonicalizing SMILES...")
        
        canonical_smiles = []
        for smiles in self.df['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        
        self.df['canonical_smiles'] = canonical_smiles
        
        # Find duplicates
        duplicates = self.df[self.df.duplicated(subset=['canonical_smiles'], keep=False)]
        self.log(f"Found {len(duplicates)} duplicate entries")
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['canonical_smiles'], keep='first')
        self.duplicates = duplicates
        return self.df
    
    def analyze_class_distribution(self):
        """
        Analyze the distribution of ADR classes in the dataset
        
        Returns:
            pd.DataFrame: Statistics for each ADR class
        """
        self.log("Analyzing class distribution...")
        
        adr_columns = [col for col in self.df.columns if col not in ['smiles', 'canonical_smiles']]
        
        self.class_stats = pd.DataFrame({
            'ADR': adr_columns,
            'Positive_Count': [self.df[col].sum() for col in adr_columns],
            'Negative_Count': [len(self.df) - self.df[col].sum() for col in adr_columns],
            'Positive_Ratio': [self.df[col].sum() / len(self.df) for col in adr_columns]
        }).sort_values('Positive_Ratio')
        
        return self.class_stats
    
    def add_molecular_properties(self):
        """
        Add basic molecular properties to the dataset
        
        Returns:
            pd.DataFrame: Dataset with molecular properties added
        """
        self.log("Computing molecular properties...")
        
        mols = [Chem.MolFromSmiles(s) for s in self.df['canonical_smiles']]
        self.df['mol_weight'] = [Descriptors.MolWt(m) for m in mols]
        self.df['num_heavy_atoms'] = [m.GetNumHeavyAtoms() for m in mols]
        self.df['num_rings'] = [Descriptors.RingCount(m) for m in mols]
        
        return self.df
    
    def create_visualizations(self):
        """
        Create preprocessing summary visualizations and save to file
        """
        self.log("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution
        ax1 = axes[0, 0]
        self.class_stats.plot(x='ADR', y='Positive_Ratio', kind='barh', ax=ax1)
        ax1.set_xlabel('Positive Class Ratio')
        ax1.set_title('ADR Class Distribution')
        
        # Molecular weight distribution
        ax2 = axes[0, 1]
        ax2.hist(self.df['mol_weight'], bins=50, edgecolor='black')
        ax2.set_xlabel('Molecular Weight')
        ax2.set_ylabel('Count')
        ax2.set_title('Molecular Weight Distribution')
        
        # Number of heavy atoms
        ax3 = axes[1, 0]
        ax3.hist(self.df['num_heavy_atoms'], bins=30, edgecolor='black')
        ax3.set_xlabel('Number of Heavy Atoms')
        ax3.set_ylabel('Count')
        ax3.set_title('Heavy Atom Count Distribution')
        
        # Top 10 imbalanced classes
        ax4 = axes[1, 1]
        top5 = self.class_stats.nlargest(5, 'Positive_Ratio')
        bottom5 = self.class_stats.nsmallest(5, 'Positive_Ratio')
        combined = pd.concat([top5, bottom5])
        combined.plot(x='ADR', y='Positive_Ratio', kind='barh', ax=ax4)
        ax4.set_xlabel('Positive Class Ratio')
        ax4.set_title('Most and Least Common ADRs')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sider_preprocessing_summary.png'), dpi=300)
        plt.close()
    
    def save_results(self):
        """
        Save processed data, reports, and statistics to files
        """
        self.log("Saving results...")
        
        # Reorder columns
        metadata_cols = ['canonical_smiles', 'mol_weight', 'num_heavy_atoms', 'num_rings']
        adr_cols = [col for col in self.df.columns if col not in metadata_cols + ['smiles']]
        final_df = self.df[metadata_cols + adr_cols]
        
        # Save cleaned data
        output_path = os.path.join(self.output_dir, 'sider_cleaned.csv')
        final_df.to_csv(output_path, index=False)
        self.log(f"Saved cleaned data to {output_path}")
        
        # Save preprocessing report
        report_path = os.path.join(self.output_dir, 'sider_preprocessing_report.txt')
        with open(report_path, 'w') as f:
            f.write("SIDER Dataset Preprocessing Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(self.report))
            f.write("\n\nClass Distribution Summary:\n")
            f.write(self.class_stats.to_string())
        
        # Save class statistics
        stats_path = os.path.join(self.output_dir, 'sider_class_stats.csv')
        self.class_stats.to_csv(stats_path, index=False)
    
    def run_pipeline(self):
        """
        Run the complete preprocessing pipeline
        
        Returns:
            pd.DataFrame: Final processed dataset
        """
        self.load_data()
        self.validate_smiles()
        self.canonicalize_and_deduplicate()
        self.analyze_class_distribution()
        self.add_molecular_properties()
        self.create_visualizations()
        self.save_results()
        self.log("Preprocessing complete!")
        return self.df

if __name__ == "__main__":
    # Run preprocessing
    preprocessor = SIDERPreprocessor(
        input_path='data/raw/sider.csv',
        output_dir='data/processed/'
    )
    preprocessor.run_pipeline()
