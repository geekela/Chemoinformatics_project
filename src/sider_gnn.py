
# CREATE THE SIDER GRAPH DATASET
# Converts each SMILES string into a graph object that PyTorch Geometric can use.
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
from sklearn.metrics import roc_auc_score

class SIDERGraphDataset(InMemoryDataset):
    """Creates a PyTorch Geometric dataset from the SIDER dataframe."""
    def __init__(self, root, df, transform=None):
        self.df = df
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            smiles = row['canonical_smiles']
            labels = row.iloc[1:28].values # Assumes labels are in columns 1 to 27

            # Convert SMILES to graph representation using OGB's utility
            graph = smiles2graph(smiles)

            # Create a PyG Data object
            data = Data(
                x=torch.tensor(graph['node_feat'], dtype=torch.long),
                edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(graph['edge_feat'], dtype=torch.long),
                y=torch.tensor(labels.astype(np.float32), dtype=torch.float).unsqueeze(0)
            )

            data_list.append(data)

        # Save the processed dataset
        torch.save(self.collate(data_list), self.processed_paths[0])

class MPNN_SIDER(pl.LightningModule):
    def __init__(self, hidden_dim, out_dim, lr=1e-3):
        super().__init__()
        self.lr = lr

        # --- Model Architecture ---
        self.atom_emb = AtomEncoder(emb_dim=hidden_dim)
        self.bond_emb = BondEncoder(emb_dim=hidden_dim)
        nn = MLP([hidden_dim, 2 * hidden_dim, hidden_dim * hidden_dim])
        self.conv = NNConv(hidden_dim, hidden_dim, nn, aggr='mean')
        self.gru = GRU(hidden_dim, hidden_dim)
        self.mlp = MLP([hidden_dim, hidden_dim // 2, out_dim])

        # --- Lists to store outputs for epoch-level metrics ---
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, data):
        # (forward pass is unchanged)
        x = self.atom_emb(data.x)
        h = x.unsqueeze(0)
        edge_attr = self.bond_emb(data.edge_attr)
        for i in range(3):
            m = F.relu(self.conv(x, data.edge_index, edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = x.squeeze(0)
        x = global_add_pool(x, data.batch)
        x = self.mlp(x)
        return x

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        preds = torch.sigmoid(out)
        self.train_outputs.append({"preds": preds, "labels": batch.y})
        return loss

    def on_train_epoch_end(self):
        # Calculate and log training AUC at the end of the epoch
        all_preds = torch.cat([x['preds'] for x in self.train_outputs]).cpu().detach().numpy()
        all_labels = torch.cat([x['labels'] for x in self.train_outputs]).cpu().detach().numpy()

        try:
            auc_macro = roc_auc_score(all_labels, all_preds, average='macro')
            self.log("train_roc_auc", auc_macro, prog_bar=True)
        except ValueError:
            self.log("train_roc_auc", 0.5, prog_bar=True) # Log 0.5 if AUC fails

        self.train_outputs.clear() # Free memory

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        preds = torch.sigmoid(out)
        self.val_outputs.append({"preds": preds, "labels": batch.y})

    def on_validation_epoch_end(self):
        # Calculate and log validation AUC at the end of the epoch
        all_preds = torch.cat([x['preds'] for x in self.val_outputs]).cpu().detach().numpy()
        all_labels = torch.cat([x['labels'] for x in self.val_outputs]).cpu().detach().numpy()

        try:
            auc_macro = roc_auc_score(all_labels, all_preds, average='macro')
            self.log("val_roc_auc", auc_macro, prog_bar=True)
        except ValueError:
            self.log("val_roc_auc", 0.5, prog_bar=True)

        self.val_outputs.clear()

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        preds = torch.sigmoid(out)
        self.test_outputs.append({"preds": preds, "labels": batch.y})

    def on_test_epoch_end(self):
        # Calculate final test score
        all_preds = torch.cat([x['preds'] for x in self.test_outputs]).cpu().detach().numpy()
        all_labels = torch.cat([x['labels'] for x in self.test_outputs]).cpu().detach().numpy()

        try:
            auc_macro = roc_auc_score(all_labels, all_preds, average='macro')
            self.log("test_macro_auc", auc_macro)
            print(f"\nFinal GNN Performance: Macro AUC-ROC = {auc_macro:.4f}")
        except ValueError as e:
            print(f"\nCould not calculate final AUC-ROC on test set: {e}")

        self.test_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
