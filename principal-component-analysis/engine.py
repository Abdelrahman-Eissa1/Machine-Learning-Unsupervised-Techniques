import numpy as np

def compute_covariance_matrix(X_std):
    """
    Computes the covariance matrix from scratch.
    Formula: C = (1 / (n-1)) * (X^T @ X)
    """
    n = X_std.shape[0]
    return (X_std.T @ X_std) / (n - 1)

def get_pca_projection_matrix(eig_vals, eig_vecs, k=2):
    """
    Sorts eigenpairs and selects the top k eigenvectors to form 
    the transformation matrix W.
    """
    # Create (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    
    # Sort tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Construct projection matrix W
    matrix_w = np.hstack([eig_pairs[i][1].reshape(-1, 1) for i in range(k)])
    return matrix_w, eig_pairs

def compute_explained_variance(eig_vals):
    """
    Calculates the individual and cumulative explained variance ratios.
    """
    total_var = sum(eig_vals)
    var_exp = [(i / total_var) for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    return var_exp, cum_var_exp
