import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import numpy as np


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
    """
    Multi-Layer Perceptron for SIDER multi-label classification.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        out_dim: Number of output labels (27 for SIDER)
        dropout: Dropout probability for regularization
        lr: Learning rate
    """

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
