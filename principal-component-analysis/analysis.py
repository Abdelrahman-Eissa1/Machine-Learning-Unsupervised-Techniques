import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import os
from engine import *

# Setup results directory
os.makedirs("results", exist_ok=True)

# 1. Load and Preprocess Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, names=columns)

X = df.iloc[:, 0:4].values
y = df['class'].values

# Standardize features (Mean=0, Var=1)
X_std = StandardScaler().fit_transform(X)

# 2. Manual PCA Implementation
cov_mat = compute_covariance_matrix(X_std)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
W, sorted_pairs = get_pca_projection_matrix(eig_vals, eig_vecs, k=2)
Y_manual = X_std @ W

# 3. Sklearn PCA (for benchmarking)
pca_sklearn = sklearnPCA(n_components=2)
Y_sklearn = pca_sklearn.fit_transform(X_std)

# 4. Explained Variance Analysis
var_exp, _ = compute_explained_variance(eig_vals)
for i, v in enumerate(var_exp):
    print(f"PC{i+1} Explained Variance: {v:.2%}")

# 5. Visualization
plt.figure(figsize=(10, 6))
for label in np.unique(y):
    plt.scatter(Y_manual[y == label, 0], Y_manual[y == label, 1], label=label)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Manual Implementation (Iris Dataset)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/pca_projection.png")
plt.show()

print("Analysis complete. Results saved in results/folder.")
