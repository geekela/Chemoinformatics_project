import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import numpy as np

class MLP_BACE(pl.LightningModule):
    """
    Multi-Layer Perceptron for BAEC regression.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        out_dim: 1 predicted value 
        dropout: Dropout probability for regularization
        lr: Learning rate
    """

    def __init__(self, input_dim, hidden_dims=[512, 256, 128], out_dim=1, dropout=0.3, lr=0.001):
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

        # Output layer
        layers.append(nn.Linear(prev_dim, out_dim))

        self.network = nn.Sequential(*layers)

        # Loss function for regression
        self.criterion = nn.MSELoss()

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
        return {'preds': logits, 'targets': y}

    def test_epoch_end(self, outputs):
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        # Calculate Root Mean Squared Error (RMSE)
        loss_fn = nn.MSELoss() 
        rmse = torch.sqrt(loss_fn(all_preds, all_targets))
        
        self.log('test_rmse', rmse, prog_bar=True)
        return {'test_rmse': rmse}

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
