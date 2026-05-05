import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

def compress_and_reconstruct(data, threshold):
    """
    Standardizes data, applies PCA based on a variance threshold, 
    and reconstructs the signal back to original space.
    """
    # 1. Standardize (Crucial for image contrast)
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    
    # 2. Apply PCA
    pca = PCA(n_components=threshold)
    compressed = pca.fit_transform(data_std)
    
    # 3. Reconstruct and Un-standardize
    recon_std = pca.inverse_transform(compressed)
    recon = scaler.inverse_transform(recon_std)
    
    return recon, pca.n_components_

def run_kernel_pca(data, kernel_type='rbf', g=0.25, deg=3):
    """
    Applies Kernel PCA to handle non-linear manifolds.
    """
    kpca = KernelPCA(kernel=kernel_type, gamma=g, degree=deg, n_components=2)
    return kpca.fit_transform(data)
