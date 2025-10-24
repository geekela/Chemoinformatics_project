import pandas as pd
import numpy as np
from typing import Union
import deepchem as dc
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import rdFingerprintGenerator, Descriptors

def featurizer(df: pd.DataFrame,methods: list=None, mol_col: str = 'Molecule', smiles: str= 'canonical_smiles', rdkit_all: bool =True,fpSize: int = 2048, concatenate: bool = False) -> Union[pd.DataFrame, dict]:
    """
    Calculates Morgan Fingerprints (2048 bits) and the specified RDKit descriptors
    using the MolecularDescriptorCalculator, adds them to the DataFrame, and cleans up.
    """

    
    if methods is None:
        methods = ["rdkit", "maccs", "MorganFP"]
    # Descriptors List
    descriptors = [
        'MolWt',
        'TPSA',
        'MolLogP',
        'NumHAcceptors',
        'NumHDonors',
        'RingCount',
        'NumAromaticHeterocycles'
    ]

    all_features = {}
    cols_orig = df.columns.tolist()
    smiles=df[smiles].tolist()
    for method in methods:
        if method == "rdkit":
            df_rd=df.copy()
            if rdkit_all==False:
                calculator = MolecularDescriptorCalculator(descriptors)
                def calc_features(mol):
                    if mol is None:
                        return [np.nan] * len(descriptors)
                    return list(calculator.CalcDescriptors(mol))
                properties_series = df[mol_col].apply(calc_features)
                df_rdkit_features = pd.DataFrame(properties_series.tolist(), columns=descriptors, index=df.index)
            else:
                featurizer = dc.feat.RDKitDescriptors()
                rdkit_feat = featurizer.featurize(smiles)
                feature_names = [f"RDKit_{i}" for i in range(rdkit_feat.shape[1])]
                df_rdkit_features = pd.DataFrame(rdkit_feat, columns=feature_names)
                     
            df_feat=pd.concat([df_rd, df_rdkit_features], axis=1)
            nan_col_list=df_feat.columns[df_feat.isna().any()].tolist()
            df_feat=df_feat.drop(columns=nan_col_list)
            all_features[method] = df_feat
            print(f"df with RDKit features shape: {df_feat.shape}")
            
        elif method == "maccs":
            df_mc=df.copy()
            featurizer = dc.feat.MACCSKeysFingerprint()
            maccs_feat = featurizer.featurize(smiles)
            feature_names = [f"MACCS_{i}" for i in range(maccs_feat.shape[1])]
            df_maccs_features = pd.DataFrame(maccs_feat, columns=feature_names)
            df_feat=pd.concat([df, df_maccs_features], axis=1)
            nan_col_list=df_feat.columns[df_feat.isna().any()].tolist()
            df_feat=df_feat.drop(columns=nan_col_list)
            all_features[method] = df_feat
            print(f"df with MACCS features shape: {df_feat.shape}")
            
        elif method == "MorganFP":
            # MORGAN FINGERPRINTS
            df_mf=df.copy()
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fpSize)

            df_mf['FingerPrint_Obj'] = df_mf[mol_col].apply(
                lambda mol: mfpgen.GetFingerprint(mol) if mol is not None else None
            )

            # Explode the BitVect column into 2048 binary columns (0s and 1s)
            # Create the list of 0s and 1s for each molecule
            fp_arrays = [
                fp.ToList() if fp is not None else [0] * fpSize
                for fp in df_mf['FingerPrint_Obj']
            ]
            fp_names = [f'Morgan_{i}' for i in range(fpSize)]
            df_mf_features = pd.DataFrame(np.array(fp_arrays), columns=fp_names, index=df_mf.index)
            df_feat=pd.concat([df_mf, df_mf_features], axis=1)
            print(f"df with Morgan Fingerprint features shape: {df_feat.shape}")
            all_features[method] = df_feat
            nan_col_list=df_feat.columns[df_feat.isna().any()].tolist()
            df_feat=df_feat.drop(columns=nan_col_list)

        else:
             raise ValueError(f"Unknown featurizer: {method}")


    # Apply the calculator to the 'Molecule' column


    # Convert the Series of lists into a new DataFrame with named columns

    """df = pd.concat([df, df_rdkit_features], axis=1)

    # MORGAN FINGERPRINTS
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fpSize)

    df['FingerPrint_Obj'] = df[mol_col].apply(
        lambda mol: mfpgen.GetFingerprint(mol) if mol is not None else None
    )

    # Explode the BitVect column into 2048 binary columns (0s and 1s)
    # Create the list of 0s and 1s for each molecule
    fp_arrays = [
        fp.ToList() if fp is not None else [0] * fpSize
        for fp in df['FingerPrint_Obj']
    ]

    fp_names = [f'Morgan_{i}' for i in range(fpSize)]
    df_fp_features = pd.DataFrame(np.array(fp_arrays), columns=fp_names, index=df.index)

    # Concatenate the new FP features to the main DataFrame
    df = pd.concat([df, df_fp_features], axis=1)
    col_to_move = df.pop('Molecule')
    df.insert(1, 'Molecule', col_to_move)"""
    
    if concatenate:
        df_concat = df[cols_orig].copy()

        for feat_df in all_features.values():
            df_concat = pd.concat([df_concat, feat_df.drop(columns=cols_orig,errors='ignore')],axis=1)
        print(f"Feature engineering complete. Final DataFrame shape: {df_concat.shape}")
        print(f"Total features added: {df_rdkit_features.shape[1]} rdkit+ {df_maccs_features.shape[1]} maccs + {df_mf_features.shape[1]} Morgan")
        return df_concat
    
    else:
        return all_features