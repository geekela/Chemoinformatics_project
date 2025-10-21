
import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from Classification.src.sider_gnn import SIDERGraphDataset, MPNN_SIDER

def objective(trial, train_loader, val_loader):

    params = {
        'lr': trial.suggest_float('lr', 1e-4, 2e-3, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128])
    }

    model = MPNN_SIDER(
        hidden_dim=params['hidden_dim'],
        out_dim=27,
        lr=params['lr']
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    wandb_logger = WandbLogger(
        project="SIDER_GNN_Bayesian_Opt",
        name=f"trial_{trial.number}",
        entity="billel-130-universit-paris-dauphine-psl"
    )

    trainer = pl.Trainer(
        max_epochs=30,
        callbacks=[early_stopping_callback],
        logger=wandb_logger,
        enable_checkpointing=False,
        accelerator="auto"
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_val_loss = trainer.callback_metrics.get("val_loss", float('inf'))

    return final_val_loss
