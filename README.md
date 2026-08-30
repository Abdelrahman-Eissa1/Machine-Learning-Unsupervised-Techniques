# Machine-Learning-Unsupervised-Techniques

A modular Python library of unsupervised machine learning implementations, covering statistical foundations, clustering theory, and dimensionality reduction from scratch using NumPy and Scikit-Learn. Based on the curriculum by Sepp Hochreiter.

## 🛠 Project Modules

### 📐 [1. Estimation Theory & Maximum Likelihood](./estimation-theory)
*   **Engineered** a modular statistical engine to compute Log-Likelihoods for Gaussian distributions, ensuring numerical stability.
*   **Quantified** estimator precision by implementing Fisher Information and the Cramér-Rao Lower Bound (CRLB).
*   **Validated** the efficiency of the arithmetic mean as an unbiased estimator through stochastic simulation.
*   **Analyzed** the impact of high-variance noise ($\sigma$) on parameter convergence and likelihood surface geometry.

### 📈 [2. Poisson Likelihood Optimization](./poisson-likelihood-analysis)
*   **Derived** the analytical Maximum Likelihood Estimator (MLE) for Poisson-distributed stochastic processes.
*   **Implemented** a numerically stable log-likelihood function using log-factorial approximations for discrete distributions.
*   **Benchmarked** analytical solutions against numerical grid-search optimization to verify mathematical convergence.
*   **Visualized** concave log-likelihood surfaces to identify global maxima and estimator sensitivity.

### 🧬 [3. Principal Component Analysis](./principal-component-analysis)
*   **Deconstructed** high-dimensional datasets into orthogonal principal components using Eigen-decomposition and SVD.
*   **Demonstrated** linear separation and feature importance ranking on multivariate datasets (Iris).
*   **Visualized** variance distribution across components to identify optimal dimensionality reduction thresholds.

### 🖼️ [4. PCA Compression & Kernel Manifolds](./pca-kernel-compression)
*   **Eigenface Pipeline:** Engineered an image compression-reconstruction pipeline from scratch. Demonstrated how 50% variance captures global structure (lighting/head shape) while 99% variance captures high-fidelity details (glasses/textures).
*   **The Kernel Trick:** Implemented **Kernel PCA** (RBF & Polynomial) to resolve non-linear manifolds. Successfully "unrolled" concentric circular data into a linearly separable 1D space.
*   **Mathematical Proofs:** Mathematically proved that the covariance matrix $C = \frac{1}{n}X^TX$ is **Positive Semi-Definite**, ensuring non-negative eigenvalues ($\lambda \ge 0$) and a physically meaningful ranking of components.

## 🧰 Tech Stack
*   **Languages:** Python 3.x
*   **Scientific Computing:** NumPy, SciPy, Pandas
*   **Machine Learning:** Scikit-Learn (Metrics & Evaluation)
*   **Data Visualization:** Matplotlib, Seaborn

## 📂 Repository Structure
To maintain a professional library format, each module is organized as follows:
*   `engine.py`: Core mathematical logic, class wrappers, and algorithm implementation.
*   `analysis.py`: Script for data generation, experimentation, and visualization.
*   `results/`: Exported graphical analysis (PNGs) and performance metrics.

---
**Author:** Abdelrahman Eissa – Bachelor of Artificial Intelligence
