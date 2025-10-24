# Molecular Property Prediction: Classification vs Regression

A comprehensive comparative study of classical machine learning and deep learning approaches for molecular property prediction on small datasets.


## Project Overview

This project systematically compares molecular property prediction approaches on two MoleculeNet benchmark datasets:
- **SIDER**: Multi-label classification of 27 drug side effects 
- **BACE**: Regression of β-secretase 1 binding affinity 

We evaluated both **classical machine learning** (Random Forest, SVM, XGBoost, Logistic Regression) and **deep learning** (MLP, GNN) models using multiple molecular representations to identify best practices for small molecular datasets.

## Key Research Questions

1. How do different molecular featurizations impact prediction performance?
2. What are the trade-offs between classical and deep learning on datasets under 2,000 molecules?
3. Does combining multiple feature representations improve results?

## Project Structure

```
Chemoinformatics_project/
├── Classification/
│   ├── SIDER_analysis.ipynb        # Complete SIDER classification pipeline
│   └── src/
│       ├── sider_preprocessing.py   # Data cleaning and canonicalization
│       ├── sider_featurizer.py      # Molecular featurization
│       ├── sider_baseline_models.py # Classical ML models
│       ├── MLP.py                   # Multi-layer perceptron
│       ├── sider_gnn.py             # Graph neural network
│       ├── sider_svm_optimization.py # SVM hyperparameter tuning
│       ├── sider_umap.py            # Dimensionality reduction visualization
│       └── chemberta_embeddings.py  # Transformer-based embeddings
│
├── Regression/
│   ├── BACE_regression.ipynb        # Complete BACE regression pipeline
│   └── src/
│       ├── featurizers.py           # Multi-featurization pipeline
│       ├── baseline_regression_models.py # Classical regressors
│       ├── MLP.py                   # Neural network for regression
│       └── bace_gnn.py              # GNN for molecular graphs
│
├── data/
│   └── raw/
│       ├── sider.csv                # SIDER dataset (1,427 molecules)
│       └── bace.csv                 # BACE dataset (1,513 molecules)
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Datasets

### SIDER (Side Effect Resource)
- **Task**: Multi-label classification
- **Labels**: 27 binary side effect categories
- **Source**: MoleculeNet

### BACE (β-secretase 1)
- **Task**: Regression
- **Target**: pIC50 binding affinity values 
- **Source**: MoleculeNet

## Molecular Featurizations

We systematically compared multiple molecular representations:

1. **Morgan Fingerprints (ECFP)**: 2,048-bit circular fingerprints (radius=2)
2. **RDKit Descriptors**: 192 physicochemical properties
3. **MACCS Keys**: 167 predefined structural patterns
4. **Combined Features**: Concatenation of all three (2,407 dimensions)
5. **ChemBERTa Embeddings**: 768-dimensional transformer-based representations

## Models Implemented

### Classical Machine Learning
- **Random Forest**: GridSearchCV for hyperparameter optimization
- **Support Vector Machines**: Linear and RBF kernels with RandomizedSearchCV
- **XGBoost**: Gradient boosting with default parameters
- **Logistic Regression**: For multi-label classification

### Deep Learning
- **Multi-Layer Perceptron (MLP)**:
  - 3 hidden layers (512, 256, 128 neurons)
  - Batch normalization and dropout
  - Early stopping with validation monitoring

- **Graph Neural Networks (GNN)**:
  - Message passing with 3 iterations
  - Node/edge feature embeddings
  - Global mean pooling
  - Bayesian optimization via Optuna (SIDER)

## Getting Started

### Prerequisites

```bash
python >= 3.8
pip install -r requirements.txt
```

### Key Dependencies
- `rdkit`: Cheminformatics toolkit
- `scikit-learn`: Classical ML algorithms
- `pytorch`: Deep learning framework
- `pytorch-geometric`: Graph neural networks
- `deepchem`: Molecular ML utilities
- `transformers`: ChemBERTa embeddings
- `optuna`: Bayesian optimization
- `umap-learn`: Dimensionality reduction

### Running the Notebooks

**SIDER Classification:**
```bash
jupyter notebook Classification/SIDER_analysis.ipynb
```

**BACE Regression:**
```bash
jupyter notebook Regression/BACE_regression.ipynb
```

## Key Visualizations

The project includes comprehensive visualizations:

1. **UMAP Projections**: Reveal structural separability for different side effects
   - Clear clustering for predictable labels (e.g., gastrointestinal disorders)
   - Scattered distribution for challenging labels (e.g., product issues)

2. **Training Dynamics**: Monitor validation vs test loss evolution
   - Detect overfitting in combined features
   - Compare convergence across featurizations

3. **Class Distribution Analysis**: Visualize severe imbalance in SIDER dataset

## Recommendations 

Based on our findings, we recommend:

1. **Establish baselines**: Start with Random Forest and SVR across multiple featurizations
2. **Use cross-validation**: Single splits show high variance; 5-10 fold CV more reliable
3. **Test feature combinations**: Benefits are task-dependent
4. **Monitor multiple metrics**: Track both validation and test performance
5. **Validate preprocessing**: Check for data leakage in normalization
6. **Use UMAP visualization**: Understand feature quality and separability
7. **Select appropriate metrics**: Use macro-averaging for severe class imbalance

## Future Work

1. Evaluate on larger datasets (>5,000 molecules) to identify regimes where learned representations surpass engineered features
2. Incorporate 3D conformational information for structure-based prediction
3. Implement uncertainty quantification for reliable confidence intervals
4. Explore ensemble methods combining classical and deep learning
5. Apply techniques to additional MoleculeNet datasets

## Resources

- **Datasets**: MoleculeNet benchmark datasets (SIDER and BACE)
- **Tools**: RDKit, PyTorch Geometric, DeepChem, scikit-learn
- **AI Assistance**: ChatGPT and Claude Code for debugging and optimization
