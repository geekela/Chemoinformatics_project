"""
Graph Neural Network for BACE Regression Task
==============================================

This module implements a GNN for predicting BACE inhibition (pIC50 values).
Similar to the reference ESOL implementation, this is a REGRESSION task.

Key differences from SIDER GNN (classification):
- Output: Single value (pIC50) instead of 27 binary labels
- Loss: MSE instead of Binary Cross Entropy
- Metric: RMSE instead of ROC-AUC
- Normalization: Yes (targets normalized to mean=0, std=1)
"""

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import GRU
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import NNConv, MLP, global_add_pool
from ogb.utils import smiles2graph
import pytorch_lightning as pl
import numpy as np
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class BACEGraphDataset(InMemoryDataset):
    """
    Creates a PyTorch Geometric dataset from the BACE dataframe.

    This follows the same pattern as:
    - ESOLGraphData (reference code for regression)
    - SIDERGraphDataset (your classification code)
    """
    def __init__(self, root, df, smiles_col='mol', target_col='pIC50', transform=None):
        """
        Args:
            root: Directory to store processed data
            df: DataFrame with SMILES and pIC50 values
            smiles_col: Name of SMILES column (default: 'mol')
            target_col: Name of target column (default: 'pIC50')
        """
        self.df = df
        self.smiles_col = smiles_col
        self.target_col = target_col
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        """Convert SMILES to graph objects"""
        data_list = []

        print(f"Converting {len(self.df)} molecules to graphs...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            smiles = row[self.smiles_col]
            target = row[self.target_col]

            try:
                # Convert SMILES to graph using OGB's utility
                graph = smiles2graph(smiles)

                # Create PyG Data object
                data = Data(
                    x=torch.tensor(graph['node_feat'], dtype=torch.long),
                    edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(graph['edge_feat'], dtype=torch.long),
                    y=torch.tensor([target], dtype=torch.float)  # Single regression value
                )

                data_list.append(data)

            except Exception as e:
                print(f"Warning: Skipping molecule due to error: {e}")
                continue

        print(f"Successfully converted {len(data_list)} molecules")

        # Save processed dataset
        torch.save(self.collate(data_list), self.processed_paths[0])


class MPNN_BACE(pl.LightningModule):
    """
    Message Passing Neural Network for BACE pIC50 prediction.

    Architecture identical to reference MPNN for ESOL regression:
    - AtomEncoder + BondEncoder (pretrained from OGB)
    - NNConv message passing with GRU for 3 iterations
    - Global add pooling
    - MLP prediction head

    Key difference from MPNN_SIDER (classification):
    - Output dimension: 1 (regression) instead of 27 (multi-label)
    - Loss: MSE instead of Binary Cross Entropy
    - No sigmoid activation (predicting continuous values)
    - Target normalization used (stored std for denormalization)
    """

    def __init__(self, hidden_dim=64, out_dim=1, std=1.0, lr=1e-3):
        """
        Args:
            hidden_dim: Embedding dimension (default: 64)
            out_dim: Output dimension (always 1 for single-target regression)
            std: Standard deviation of normalized targets (for denormalization)
            lr: Learning rate (default: 1e-3)
        """
        super().__init__()
        self.std = std  # Store for denormalizing predictions
        self.lr = lr

        # --- Model Architecture (identical to reference) ---
        # Initial embeddings
        self.atom_emb = AtomEncoder(emb_dim=hidden_dim)
        self.bond_emb = BondEncoder(emb_dim=hidden_dim)

        # Message passing layers
        nn = MLP([hidden_dim, 2 * hidden_dim, hidden_dim * hidden_dim])
        self.conv = NNConv(hidden_dim, hidden_dim, nn, aggr='mean')
        self.gru = GRU(hidden_dim, hidden_dim)

        # Readout layers
        self.mlp = MLP([hidden_dim, hidden_dim // 2, out_dim])

        # Lists to store outputs for epoch-level metrics
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, data):
        """
        Forward pass through the GNN.

        Returns predictions in NORMALIZED space (for training loss).
        """
        # Initialization
        x = self.atom_emb(data.x)
        h = x.unsqueeze(0)
        edge_attr = self.bond_emb(data.edge_attr)

        # Message passing (3 iterations)
        for i in range(3):
            m = F.relu(self.conv(x, data.edge_index, edge_attr))  # Message + aggregation
            x, h = self.gru(m.unsqueeze(0), h)  # Node update
            x = x.squeeze(0)

        # Readout
        x = global_add_pool(x, data.batch)  # Graph-level embedding
        x = self.mlp(x)  # Prediction

        return x.view(-1)  # Flatten to [batch_size] for MSE loss

    def training_step(self, batch, batch_idx):
        """Training step (computed on normalized values)"""
        out = self.forward(batch)
        loss = F.mse_loss(out, batch.y)  # MSE on normalized targets

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=batch.num_graphs)

        # Store for epoch-level metrics
        self.train_outputs.append({"preds": out, "labels": batch.y})
        return loss

    def on_train_epoch_end(self):
        """Calculate and log RMSE at the end of training epoch"""
        all_preds = torch.cat([x['preds'] for x in self.train_outputs]).cpu().detach()
        all_labels = torch.cat([x['labels'] for x in self.train_outputs]).cpu().detach()

        # Calculate RMSE in ORIGINAL space (denormalized)
        rmse = torch.sqrt(F.mse_loss(all_preds * self.std, all_labels * self.std))
        self.log("train_rmse", rmse, prog_bar=True)

        self.train_outputs.clear()  # Free memory

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        out = self.forward(batch)

        # Loss on normalized targets
        loss = F.mse_loss(out, batch.y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch.num_graphs)

        # Store for epoch-level metrics
        self.val_outputs.append({"preds": out, "labels": batch.y})

    def on_validation_epoch_end(self):
        """Calculate and log RMSE on validation set"""
        all_preds = torch.cat([x['preds'] for x in self.val_outputs]).cpu().detach()
        all_labels = torch.cat([x['labels'] for x in self.val_outputs]).cpu().detach()

        # RMSE in original space
        rmse = torch.sqrt(F.mse_loss(all_preds * self.std, all_labels * self.std))
        self.log("val_rmse", rmse, prog_bar=True)

        self.val_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step"""
        out = self.forward(batch)
        self.test_outputs.append({"preds": out, "labels": batch.y})

    def on_test_epoch_end(self):
        """Calculate final test performance"""
        all_preds = torch.cat([x['preds'] for x in self.test_outputs]).cpu().detach()
        all_labels = torch.cat([x['labels'] for x in self.test_outputs]).cpu().detach()

        # Calculate metrics in original space
        mse = F.mse_loss(all_preds * self.std, all_labels * self.std)
        rmse = torch.sqrt(mse)

        self.log("test_mse", mse)
        self.log("test_rmse", rmse)

        print(f"\n{'='*70}")
        print(f"Final GNN Performance on BACE Test Set")
        print(f"{'='*70}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"{'='*70}\n")

        self.test_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer (Adam)"""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
