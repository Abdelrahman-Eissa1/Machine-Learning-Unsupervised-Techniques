import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from engine import compress_and_reconstruct, run_kernel_pca

# Create results folder
os.makedirs("results", exist_ok=True)

# --- PART 1: Face Image Compression (faces94.csv) ---
try:
    data = np.genfromtxt('faces94.csv', delimiter=',')
    thresholds = [0.50, 0.75, 0.99]
    indices = [0, 1, 2] # Images to visualize

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 12))

    for i, img_idx in enumerate(indices):
        # Plot Original
        axes[i, 0].pcolor(data[img_idx].reshape(45, 50).T, cmap=cm.gray)
        axes[i, 0].set_title("Original (2250 dims)")
        axes[i, 0].set_ylim([45, 0]) # FIX: Right-side up
        axes[i, 0].axis("off")
        
        for j, thres in enumerate(thresholds):
            recon, n_comp = compress_and_reconstruct(data, thres)
            
            axes[i, j+1].pcolor(recon[img_idx].reshape(45, 50).T, cmap=cm.gray)
            axes[i, j+1].set_title(f"Recon {int(thres*100)}%\n({n_comp} PCs)")
            axes[i, j+1].set_ylim([45, 0]) # FIX: Right-side up
            axes[i, j+1].axis("off")

    plt.tight_layout()
    plt.savefig("results/face_compression_final.png")
    plt.show()
    print("Success: Face compression plot saved.")
except Exception as e:
    print(f"Error in Face Analysis: {e}")

# --- PART 2: Kernel PCA Analysis (pca4.csv) ---
try:
    data4 = np.genfromtxt('pca4.csv', delimiter=',')
    X, labels = data4[:, :2], data4[:, 2]
    
    # Run RBF Kernel PCA
    X_kpca = run_kernel_pca(X, kernel_type='rbf', g=0.25)
    
    # Plot 1D Separation (matching your PDF)
    plt.figure(figsize=(10, 3))
    plt.scatter(X_kpca[:, 0], np.zeros_like(X_kpca[:, 0]), c=labels, cmap='viridis', alpha=0.6)
    plt.title("1D Projection via RBF-Kernel PCA (Class Separation)")
    plt.xlabel("First Principal Component")
    plt.grid(True, axis='x', alpha=0.3)
    plt.savefig("results/kpca_1d_separation.png")
    plt.show()
    print("Success: Kernel PCA plot saved.")
except Exception as e:
    print(f"Error in Kernel PCA Analysis: {e}")
