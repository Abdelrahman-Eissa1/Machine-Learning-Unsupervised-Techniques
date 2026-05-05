import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from poisson_engine import *

# Setup results directory
os.makedirs("results", exist_ok=True)

# 1. Load Data
# Note: Ensure 'poisson.csv' is in the same folder
try:
    data_df = pd.read_csv('poisson.csv', header=None)
    X = data_df.values[:, 0]
except FileNotFoundError:
    print("poisson.csv not found. Generating dummy data for demonstration...")
    X = np.random.poisson(lam=10.5, size=1000)

# 2. Compute Estimators
lam_set = np.linspace(1, 50, 99)
lam_analytical = get_analytical_mle(X)
lam_numerical, log_lik_curve = get_numerical_mle(X, lam_set)

print(f"Analytical Lambda-Hat: {lam_analytical:.3f}")
print(f"Numerical Lambda-Hat:  {lam_numerical:.3f}")

# 3. Visualization
plt.figure(figsize=(10, 6))
plt.plot(lam_set, log_lik_curve, label='Log-Likelihood Curve', color='blue')
plt.axvline(lam_analytical, color='red', linestyle='--', alpha=0.6, 
            label=f'Analytical Max: {lam_analytical:.3f}')
plt.scatter(lam_numerical, np.max(log_lik_curve), color='green', marker='x', s=100,
            label=f'Numerical Max: {lam_numerical:.3f}', zorder=5)

plt.xlabel(r'$\lambda$')
plt.ylabel('Log-Likelihood')
plt.title('Poisson Log-Likelihood Optimization')
plt.legend()
plt.grid(True, alpha=0.3)

# Save result
plt.savefig("results/poisson_mle_comparison.png", dpi=300)
plt.show()
