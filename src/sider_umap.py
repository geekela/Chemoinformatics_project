import matplotlib.pyplot as plt
import pandas as pd

def plot_umap(umap_df: pd.DataFrame, labels_df: pd.DataFrame, label_name: str):
    plot_df = umap_df.copy()
    plot_df['Target'] = labels_df[label_name].values

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        plot_df['UMAP 1'],
        plot_df['UMAP 2'],
        c=plot_df['Target'],
        cmap='viridis',
        s=10
    )

    plt.colorbar(scatter, ticks=[0, 1], label=label_name)
    plt.title(f'UMAP Projection by "{label_name}"')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(alpha=0.3)
    plt.show()
