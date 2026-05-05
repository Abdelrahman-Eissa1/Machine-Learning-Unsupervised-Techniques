import numpy as np
import matplotlib.pyplot as plt
import os
from mle_engine import *

# Setup directory for saving plots
os.makedirs("results", exist_ok=True)

def save_plot(filename):
    plt.savefig(f"results/{filename}", dpi=300, bbox_inches='tight')
    print(f"Saved: results/{filename}")

# 1. Visualization of Likelihood vs Log-Likelihood
L = np.linspace(0.001, 5, 1000)
plt.figure(figsize=(10, 5))
plt.plot(L, np.log(L), label='ln(L)')
plt.title("Monotonic Transformation: Log-Likelihood vs Likelihood")
plt.grid(True)
save_plot("likelihood_comparison.png")

# 2. Data Generation & MLE Analysis
mu_true = 5.0
sigmas = [1.0, 3.0, 10.0]
n = 500
mu_set = np.linspace(-5, 15, 501)

plt.figure(figsize=(12, 6))
for i, s in enumerate(sigmas):
    data = s * np.random.randn(n) + mu_true
    lnL_array = calculate_log_likelihood_over_mu(data, s, mu_set)
    mu_hat = mu_set[np.argmax(lnL_array)]
    
    plt.plot(mu_set, lnL_array, label=f'σ={s}, μ_hat={mu_hat:.2f}')
    print(f"Sigma: {s} | Fisher Info: {fisher_information_mu(n, s):.2f} | MLE mu: {mu_hat}")

plt.title("Log-Likelihood Curves for varying Sigma")
plt.legend()
plt.grid(True)
save_plot("log_likelihood_curves.png")
plt.show()
