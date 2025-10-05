# Chemoinformatics Project: SIDER Dataset Analysis

A comprehensive chemoinformatics project for analyzing the SIDER (Side Effect Resource) dataset, focusing on adverse drug reaction (ADR) prediction using molecular data and machine learning techniques.

## Project Overview

This project implements a complete pipeline for preprocessing and analyzing the SIDER dataset, which contains molecular structures (SMILES) and their associated adverse drug reactions. The goal is to build machine learning models that can predict potential side effects of drug molecules.

## Project Structure

```
Chemoinformatics_project/
├── data/
│   ├── raw/                    # Original datasets
│   │   └── sider.csv          # SIDER dataset (1,427 molecules, 27 ADR classes)
│   ├── processed/             # Cleaned and processed datasets
│   │   ├── sider_cleaned.csv  # Preprocessed dataset with molecular properties
│   │   ├── sider_class_stats.csv  # ADR class distribution statistics
│   │   ├── sider_preprocessing_report.txt  # Detailed processing report
│   │   ├── sider_preprocessing_summary.png  # Visualization summary
│   │   └── sider_summary_stats.csv  # Summary statistics
│   └── reports/               # Additional analysis reports
├── notebooks/                 # Jupyter notebooks for interactive analysis
│   └── 01_sider_preprocessing.ipynb  # Preprocessing and data exploration
├── src/                      # Source code modules
│   ├── preprocessing/        # Data preprocessing pipeline
│   │   └── sider_preprocessing.py  # Main preprocessing class
│   ├── featurization/        # Molecular feature extraction (future)
│   ├── models/               # Machine learning models (future)
│   └── evaluation/           # Model evaluation metrics (future)
├── results/                  # Model results and figures (future)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```


## What We Accomplished

### 1. Data Preprocessing Pipeline

Created a comprehensive preprocessing pipeline (`SIDERPreprocessor` class) that:

- **Validates SMILES strings**: Ensures all molecular structures are chemically valid
- **Canonicalizes SMILES**: Standardizes molecular representations
- **Removes duplicates**: Eliminates redundant molecules
- **Computes molecular properties**: Adds molecular weight, heavy atom count, ring count
- **Analyzes class distribution**: Provides detailed statistics on ADR class imbalance
- **Generates visualizations**: Creates summary plots and reports

### 2. Data Quality Assessment

**Dataset Statistics:**
- **Total molecules**: 1,427
- **ADR classes**: 27
- **Invalid SMILES**: 0 (100% valid)
- **Duplicates found**: 0 (clean dataset)
- **Missing values**: 0

**Class Imbalance Analysis:**
- **Most common ADR**: Skin and subcutaneous tissue disorders (92.4%)
- **Least common ADR**: Product issues (1.5%)
- **Class imbalance ratio**: ~60:1 (highly imbalanced)
- **Average positive ratio**: 56.8%

### 3. Molecular Properties Added

The preprocessing pipeline automatically computes:
- **Molecular Weight** (Da): Average ~353 Da
- **Heavy Atom Count**: Average ~25 atoms
- **Ring Count**: Average ~3 rings

### 4. Generated Outputs

**Processed Data Files:**
- `sider_cleaned.csv`: Main dataset with molecular properties
- `sider_class_stats.csv`: ADR class distribution statistics
- `sider_preprocessing_report.txt`: Detailed processing log
- `sider_preprocessing_summary.png`: Visualization summary
- `sider_summary_stats.csv`: Key statistics summary

## Technical Details

### Dependencies

The project uses the following key libraries:
- **pandas** (≥2.1.0): Data manipulation and analysis
- **numpy** (≥1.26.0): Numerical computations
- **rdkit** (≥2023.9.1): Chemical informatics toolkit
- **scikit-learn** (≥1.3.0): Machine learning algorithms
- **matplotlib** (≥3.7.2): Plotting and visualization
- **seaborn** (≥0.12.2): Statistical data visualization
- **jupyter** (≥1.0.0): Interactive notebooks
- **mordred** (≥1.2.0): Molecular descriptor calculation
- **xgboost** (≥1.7.6): Gradient boosting framework

### Key Features of the Preprocessing Pipeline

1. **Robust SMILES Validation**: Uses RDKit to ensure chemical validity
2. **Comprehensive Logging**: Tracks all preprocessing steps with timestamps
3. **Class Imbalance Analysis**: Detailed statistics on ADR distribution
4. **Molecular Property Computation**: Automatic calculation of key descriptors
5. **Visualization Generation**: Automatic creation of summary plots
6. **Report Generation**: Comprehensive text and CSV reports

## 📊 Data Insights

### Class Distribution Highlights

The SIDER dataset shows significant class imbalance:

**Most Common ADRs (Top 5):**
1. Skin and subcutaneous tissue disorders (92.4%)
2. Nervous system disorders (91.4%)
3. Gastrointestinal disorders (91.0%)
4. General disorders and administration site conditions (90.5%)
5. Investigations (80.7%)

**Least Common ADRs (Bottom 5):**
1. Product issues (1.5%)
2. Pregnancy, puerperium and perinatal conditions (8.8%)
3. Surgical and medical procedures (14.9%)
4. Social circumstances (17.6%)
5. Congenital, familial and genetic disorders (17.7%)

### Molecular Property Ranges

- **Molecular Weight**: 18 - 7,595 Da (median: 353 Da)
- **Heavy Atoms**: 1 - 492 atoms (median: 25 atoms)
- **Rings**: 0 - 46 rings (median: 3 rings)

## 🎯 Next Steps

This preprocessing pipeline sets the foundation for:

1. **Molecular Featurization**: Extract additional molecular descriptors
2. **Machine Learning Models**: Build ADR prediction models
3. **Class Imbalance Handling**: Implement techniques for imbalanced datasets
4. **Model Evaluation**: Comprehensive performance assessment
5. **Visualization**: Advanced molecular and model visualizations

## 📝 Usage Examples

### Running the Preprocessing Pipeline

```python
from src.preprocessing.sider_preprocessing import SIDERPreprocessor

# Initialize preprocessor
preprocessor = SIDERPreprocessor(
    input_path='data/raw/sider.csv',
    output_dir='data/processed/'
)

# Run complete pipeline
processed_data = preprocessor.run_pipeline()
```

### Loading Processed Data

```python
import pandas as pd

# Load cleaned dataset
cleaned_data = pd.read_csv('data/processed/sider_cleaned.csv')

# Load class statistics
class_stats = pd.read_csv('data/processed/sider_class_stats.csv')

print(f"Dataset shape: {cleaned_data.shape}")
print(f"Classes: {len(class_stats)}")
```

## 🤝 Contributing

This project is designed for chemoinformatics research and education. Feel free to:
- Extend the preprocessing pipeline
- Add new molecular descriptors
- Implement machine learning models
- Improve visualization capabilities

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

---

**Note**: The preprocessing pipeline has been tested and validated with the SIDER dataset. All molecular structures are chemically valid, and the class imbalance has been thoroughly analyzed to inform future modeling decisions.
