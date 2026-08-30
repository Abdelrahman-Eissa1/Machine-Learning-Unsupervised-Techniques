import numpy as np
import matplotlib.pyplot as plt
from engine import FactorAnalysisEngine
import os

if not os.path.exists('results'): os.makedirs('results')

def generate_toy_data(n=100, m=5, l=3):
    U_true = np.random.randint(-3, 4, size=(m, l))
    psi_true = np.random.randint(1, 4, size=m)
    Z = np.random.normal(0, 1, size=(l, n))
    Eps = np.random.multivariate_normal(np.zeros(m), np.diag(psi_true), size=n).T
    X = (U_true @ Z + Eps).T
    return X, l

def run_benchmarks():
    X, l = generate_toy_data()
    n_runs = 10
    n_steps = 100
    
    gd_all_losses = []
    em_all_losses = []

    for r in range(n_runs):
        # Shared initialization
        U_init = np.random.uniform(-0.1, 0.1, (X.shape[1], l))
        psi_init = np.random.uniform(0.01, 0.1, X.shape[1])
        
        # GD Run
        u_gd, p_gd = U_init.copy(), psi_init.copy()
        gd_l = []
        for _ in range(n_steps):
            gd_l.append(FactorAnalysisEngine.compute_loss(X, u_gd, p_gd))
            u_gd, p_gd = FactorAnalysisEngine.grad_step(X, u_gd, p_gd, 0.001)
        gd_all_losses.append(gd_l)

        # EM Run
        u_em, p_em = U_init.copy(), psi_init.copy()
        em_l = []
        for _ in range(n_steps):
            em_l.append(FactorAnalysisEngine.compute_loss(X, u_em, p_em))
            u_em, p_em = FactorAnalysisEngine.em_step(X, u_em, p_em)
        em_all_losses.append(em_l)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for i in range(n_runs):
        ax1.plot(gd_all_losses[i], alpha=0.6)
        ax2.plot(em_all_losses[i], alpha=0.6)
        
    ax1.set_title("Gradient Descent Convergence", fontsize=14)
    ax2.set_title("Expectation Maximization Convergence", fontsize=14)
    ax1.set_ylabel("Negative Log-Likelihood")
    ax1.set_xlabel("Iterations")
    ax2.set_xlabel("Iterations")
    
    plt.tight_layout()
    plt.savefig('results/gd_vs_em_convergence.png')
    plt.show()

if __name__ == "__main__":
    run_benchmarks()