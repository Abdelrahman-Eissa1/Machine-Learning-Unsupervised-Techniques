# Machine-Learning-Unsupervised-Techniques

A modular Python library of unsupervised machine learning implementations, covering statistical foundations, clustering theory, and dimensionality reduction from scratch using NumPy and Scikit-Learn.

## 🚀 Project Modules

### 📐 [01. Estimation Theory & MLE](./01-estimation-theory)
*   **Engineered** a modular statistical engine to compute Log-Likelihoods for Gaussian distributions, ensuring numerical stability.
*   **Quantified** estimator precision by implementing Fisher Information and the Cramér-Rao Lower Bound (CRLB).
*   **Validated** the efficiency of the arithmetic mean as an unbiased estimator through stochastic simulation.
*   **Analyzed** the impact of high-variance noise ($\sigma$) on parameter convergence and likelihood surface geometry.

### 🧩 [02. Latent Structure Discovery (Coming Soon)](#)
*   **Future Module:** Implementation of clustering algorithms and density-based pattern discovery.

---

## 📂 Repository Structure
This repository follows a modular architecture to demonstrate clean software engineering practices in a machine learning context:
```text
├── [module-folder]/
│   ├── engine.py           # Core logic and mathematical implementations
│   ├── analysis.py         # Script to run experiments and generate visuals
│   ├── README.md           # Module-specific documentation and results
│   └── results/            # Exported visualizations (.png)
└── README.md               # Main portfolio documentation
