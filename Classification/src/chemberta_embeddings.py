import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm

def get_chemberta_embeddings(smiles_list):
    """
    Generates embeddings for a list of SMILES strings using a pretrained ChemBERTa model.
    """
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    model.eval()

    all_embeddings = []

    for smiles in tqdm(smiles_list):
        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # Get embeddings without calculating gradients for faster inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the representation of the [CLS] token, which summarizes the entire molecule
        embedding = outputs.last_hidden_state[0, 0, :].numpy()
        all_embeddings.append(embedding)

    return np.array(all_embeddings)
