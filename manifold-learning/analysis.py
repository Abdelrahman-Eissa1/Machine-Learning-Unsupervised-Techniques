import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, FastICA
from sklearn import manifold
from engine import ManifoldEngine
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def run_manifold_comparison():
    """
    Loads the Wine dataset, runs a from-scratch t-SNE implementation,
    and compares it against standard Scikit-Learn manifold learning algorithms.
    """
    print("--- Loading and Preprocessing Data ---")
    # Load Wine dataset
    # Expected format: class label in column 0, features in columns 1-13
    try:
        df = pd.read_csv('wine.data', header=None)
    except FileNotFoundError:
        print("Error: 'wine.data' not found in the current directory.")
        return

    # Scale data: Zero mean and unit variance
    X = scale(df.values[:, 1:])
    y = df.values[:, 0]

    # 1. Execute From-scratch t-SNE (from engine.py)
    print("Running custom t-SNE optimization (this may take a moment)...")
    # Target perplexity of 50 as specified in the course material
    P = ManifoldEngine.p_joint(X, 50)
    y_tsne_custom = ManifoldEngine.estimate_tsne(X, P, iters=500, lr=10)

    # 2. Define Comparison Methods
    # Parameters updated to satisfy Scikit-Learn 1.9+ requirements
    methods = {
        "PCA": PCA(n_components=2),
        "FastICA": FastICA(n_components=2, random_state=42),
        "Isomap": manifold.Isomap(n_neighbors=20, n_components=2),
        "LLE": manifold.LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='standard'),
        "Metric MDS": manifold.MDS(
            n_components=2, 
            metric_mds=True,           # Future-proofed parameter
            n_init=4,                  # Explicit initialization count
            init='random',             # Explicit initialization method
            normalized_stress='auto',
            random_state=42
        ),
        "t-SNE (From Scratch)": None    # Placeholder for our custom result
    }

    # 3. Visualization Setup
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    print("--- Beginning Projections ---")

    for i, (name, model) in enumerate(methods.items()):
        print(f"Processing: {name}...")
        
        if name == "t-SNE (From Scratch)":
            coords = y_tsne_custom
        else:
            coords = model.fit_transform(X)
        
        # Plotting the 2D embedding
        scatter = axes[i].scatter(
            coords[:, 0], 
            coords[:, 1], 
            c=y, 
            cmap='viridis', 
            edgecolors='k', 
            s=60, 
            alpha=0.8
        )
        axes[i].set_title(name, fontsize=15, fontweight='bold')
        axes[i].axis('off')
        axes[i].grid(False)

    plt.suptitle("Manifold Learning Comparison: Dimensionality Reduction on Wine Dataset", 
                 fontsize=20, y=0.98)
    
    # Save the output for the GitHub portfolio
    output_path = 'results/manifold_comparison.png'
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"--- Analysis Complete! Visual saved to {output_path} ---")
    plt.show()

if __name__ == "__main__":
    run_manifold_comparison()