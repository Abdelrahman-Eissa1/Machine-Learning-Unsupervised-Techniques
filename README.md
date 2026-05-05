# Machine-Learning-Unsupervised-Techniques

A modular Python library of unsupervised machine learning implementations, covering statistical foundations, clustering theory, and dimensionality reduction from scratch using NumPy and Scikit-Learn.

## 🛠 Project Modules

### 📐 [Estimation Theory & Maximum Likelihood](./estimation-theory)
*   **Engineered** a modular statistical engine to compute Log-Likelihoods for Gaussian distributions, ensuring numerical stability.
*   **Quantified** estimator precision by implementing Fisher Information and the Cramér-Rao Lower Bound (CRLB).
*   **Validated** the efficiency of the arithmetic mean as an unbiased estimator through stochastic simulation.
*   **Analyzed** the impact of high-variance noise ($\sigma$) on parameter convergence and likelihood surface geometry.

### 📈 [Poisson Likelihood Optimization](./poisson-likelihood-analysis)
*   **Derived** the analytical Maximum Likelihood Estimator (MLE) for Poisson-distributed stochastic processes.
*   **Implemented** a numerically stable log-likelihood function using log-factorial approximations for discrete distributions.
*   **Benchmarked** analytical solutions against numerical grid-search optimization to verify mathematical convergence.
*   **Visualized** concave log-likelihood surfaces to identify global maxima and estimator sensitivity.

### 🖼️ [PCA Compression & Kernel Manifolds](./pca-kernel-compression)
*   **Engineered** an image compression-reconstruction pipeline from scratch, achieving significant dimensionality reduction on facial datasets.
*   **Derived** the mathematical proof for the positive semi-definiteness of covariance matrices and the necessity of non-negative eigenvalues.
*   **Implemented** Kernel PCA (RBF & Polynomial) to resolve non-linear manifolds and achieve linear class separation in 1D projections.
*   **Benchmarked** explained variance thresholds against reconstruction fidelity to quantify the information-loss trade-off.

## 🧰 Tech Stack
*   **Languages:** Python 3.x
*   **Scientific Computing:** NumPy, SciPy, Pandas
*   **Machine Learning:** Scikit-Learn (Metrics & Evaluation)
*   **Data Visualization:** Matplotlib, Seaborn

## 📂 Repository Structure
To maintain a modular library format, each module contains:
*   `engine.py`: Core mathematical logic and algorithm implementation.
*   `analysis.py`: Script for data generation, experimentation, and visualization.
*   `results/`: Exported graphical analysis and performance metrics.

---
**Author:** Abdelrahman Eissa – Bachelor of Artificial Intelligence
