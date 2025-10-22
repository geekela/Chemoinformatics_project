import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def analyze_pca_variance(df_final: pd.DataFrame):
    """
    Performs exploratory analysis using PCA on the feature set.
    1. Isolates and cleans features (removes zero-variance).
    2. Scales the features.
    3. Computes the first 10 principal components.
    4. Plots the explained variance (scree plot) and prints the cumulative variance.
    """

    features = df_final.iloc[:, 29:].copy()

    # Remove any non-numeric or object columns that might remain
    if 'FingerPrint_Obj' in features.columns:
        features.drop(columns=['FingerPrint_Obj'], inplace=True)

    #features = features.select_dtypes(include=[np.number])


    print(f"Number of features before removing zero-variance: {features.shape[1]}")

    # Remove zero-variance features
    selector = VarianceThreshold(threshold=0.0)
    features_cleaned = selector.fit_transform(features)

    print(f"Number of features after removing zero-variance: {features_cleaned.shape[1]}")

    features_df = pd.DataFrame(features_cleaned)
    features_df=features_df.dropna(axis=1)
    print(f" NA value  in features_df if any: {features_df.isnull().sum().sum()}")
    # Scaling
    x_scaled = StandardScaler().fit_transform(features_df)
    print(f"Scaled features NA value if any: {np.isnan(x_scaled).sum()}")
    
    #drop nan from scaled
    print(x_scaled.shape)
    x_scaled=x_scaled[~np.isnan(x_scaled)]

    # Compute PCA
    pca = PCA(n_components=10)
    pca.fit(x_scaled) # No need to store the transformed data (x_pca) for this analysis


    print("\n--- Principal Component Analysis (PCA) ---")

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, 11), pca.explained_variance_ratio_, marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Proportion of Explained Variance")
    plt.title("Scree Plot of Principal Components")
    plt.grid(True)
    plt.show()

    print("\nCumulative Explained Variance by the first 10 components:")
    print(np.cumsum(pca.explained_variance_ratio_))
