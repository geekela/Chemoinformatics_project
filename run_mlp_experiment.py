#!/usr/bin/env python3
"""
Script to execute the MLP experiment from SIDER_analysis notebook
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

# Import custom modules
import sys
sys.path.append('/Users/elisa/CHEMO/Chemoinformatics_project')
from Classification.src.sider_preprocessing import sider_preprocessing
from Classification.src.sider_featurizer import featurizer

print("="*80)
print("SIDER MLP EXPERIMENT")
print("="*80)

# 1. Load and preprocess data
print("\n1. Loading and preprocessing data...")
df = pd.read_csv('/Users/elisa/CHEMO/Chemoinformatics_project/data/raw/sider.csv')
print(f"   Loaded {len(df)} molecules with {len(df.columns)-1} side effect labels")

df_cleaned = sider_preprocessing(df)

# 2. Feature engineering
print("\n2. Generating features...")
df_final = featurizer(df=df_cleaned, mol_col='Molecule', fpSize=2048)

# 3. Prepare features and labels
print("\n3. Preparing features and labels...")
X = df_final.iloc[:, 29:].copy()
y = df_final.iloc[:, 2:29]

X = X.select_dtypes(include=np.number)

# Apply VarianceThreshold
selector = VarianceThreshold(threshold=0.0)
X_cleaned_array = selector.fit_transform(X)
X = pd.DataFrame(X_cleaned_array, columns=X.columns[selector.get_support()])

# 4. Split and scale data
print("\n4. Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   Data split: {len(X_train)} training, {len(X_test)} testing samples")
print(f"   Input features: {X_train_scaled.shape[1]}")
print(f"   Output labels: {y_train.shape[1]}")

# 5. Define Dataset and Model classes
class SIDERDataset(Dataset):
    """PyTorch Dataset wrapper for SIDER data."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values if hasattr(y, 'values') else y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP_SIDER(pl.LightningModule):
    """Multi-Layer Perceptron for SIDER multi-label classification."""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128], out_dim=27, dropout=0.3, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Build MLP architecture dynamically
        layers = []
        prev_dim = input_dim

        # Add hidden layers with activation, batch norm, and dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation - using BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, out_dim))

        self.network = nn.Sequential(*layers)

        # Loss function for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        return {'preds': probs, 'targets': y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# 6. Prepare data loaders
print("\n5. Preparing data loaders...")
train_dataset = SIDERDataset(X_train_scaled, y_train)
test_dataset = SIDERDataset(X_test_scaled, y_test)

# Create validation split
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 32
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"   Training samples: {len(train_subset)}")
print(f"   Validation samples: {len(val_subset)}")
print(f"   Test samples: {len(test_dataset)}")

# 7. Initialize and train MLP
print("\n6. Training MLP model...")
print("   Architecture: Input({}) → 512 → 256 → 128 → Output(27)".format(X_train_scaled.shape[1]))
print("   Dropout: 0.3, Learning rate: 0.001")

input_dim = X_train_scaled.shape[1]
mlp_model = MLP_SIDER(
    input_dim=input_dim,
    hidden_dims=[512, 256, 128],
    out_dim=27,
    dropout=0.3,
    lr=0.001
)

# Setup callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='mlp_checkpoints/',
    filename='best-mlp-model',
    save_top_k=1,
    mode='min'
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=15,
    mode='min',
    verbose=True
)

# Train
trainer = pl.Trainer(
    max_epochs=20,  # Reduced for faster testing
    callbacks=[checkpoint_callback, early_stop_callback],
    accelerator='auto',
    log_every_n_steps=10,
    enable_progress_bar=False,  # Disable progress bar for cleaner output
    enable_model_summary=False
)

print("\n   Starting training...")
trainer.fit(mlp_model, train_loader, val_loader)

# 8. Evaluate on test set
print("\n7. Evaluating MLP on test set...")
mlp_model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        logits = mlp_model(x)
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(y.cpu().numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# Calculate metrics
macro_auc = roc_auc_score(all_targets, all_preds, average='macro')
micro_auc = roc_auc_score(all_targets, all_preds, average='micro')

# Calculate per-label AUC
label_aucs = []
for i, label in enumerate(y_test.columns):
    try:
        auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
        label_aucs.append((label, auc))
    except:
        label_aucs.append((label, 0.5))

label_aucs.sort(key=lambda x: x[1], reverse=True)

# 9. Display results
print("\n" + "="*80)
print("MLP PERFORMANCE RESULTS")
print("="*80)
print(f"\nOverall Performance:")
print(f"  Macro AUC-ROC: {macro_auc:.4f}")
print(f"  Micro AUC-ROC: {micro_auc:.4f}")

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON (MACRO AUC-ROC)")
print("="*80)

results = {
    'Random Forest': 0.6691,
    'XGBoost': 0.6596,
    'GNN (default)': 0.6408,
    'MLP (NEW)': macro_auc,
    'Logistic Regression': 0.6213,
    'SVM (linear)': 0.6145,
    'Transformer + RF': 0.6070,
    'GNN (optimized)': 0.5886
}

# Sort by performance
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for model, score in sorted_results:
    marker = " ← MLP" if "MLP" in model else ""
    print(f"{model:25s} {score:.4f}{marker}")

print("\n" + "="*80)
print("TOP 5 BEST PREDICTED LABELS (MLP)")
print("="*80)
for label, auc in label_aucs[:5]:
    print(f"{label:50s} {auc:.4f}")

print("\n" + "="*80)
print("TOP 5 WORST PREDICTED LABELS (MLP)")
print("="*80)
for label, auc in label_aucs[-5:]:
    print(f"{label:50s} {auc:.4f}")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)