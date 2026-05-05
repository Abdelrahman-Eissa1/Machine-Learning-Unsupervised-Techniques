# Unsupervised Machine Learning: Implementations & Analysis

This repository is a comprehensive collection of core Unsupervised Learning algorithms implemented from the ground up, focusing on statistical foundations, pattern discovery, and dimensionality reduction.

## 🛠 Project Modules

### 📐 [Estimation Theory & Maximum Likelihood](./01-estimation-theory)
*   Engineered a modular statistical engine to compute Log-Likelihoods for Gaussian distributions, ensuring numerical stability.
*   Quantified estimator precision by implementing Fisher Information and the Cramér-Rao Lower Bound (CRLB).
*   Validated the efficiency of the arithmetic mean as an unbiased estimator through stochastic simulation.
*   Analyzed the impact of high-variance noise ($\sigma$) on parameter convergence and likelihood surface geometry.

### 🧩 [Clustering Foundations: K-Means & Beyond](./)
*   *Upcoming module focusing on centroid-based clustering and cluster stability analysis.*

### 🌌 [Dimensionality Reduction: PCA & SVD](./)
*   *Upcoming module focusing on feature extraction and high-dimensional data compression.*

### 🔍 [Density-Based & Hierarchical Clustering](./)
*   *Upcoming module focusing on DBSCAN and Agglomerative Hierarchical clustering.*

## 🧰 Tech Stack
*   **Languages:** Python 3.x
*   **Scientific Computing:** NumPy, SciPy
*   **Machine Learning:** Scikit-Learn (Metrics & Evaluation)
*   **Data Visualization:** Matplotlib, Seaborn

## 📂 Repository Structure
To maintain a modular library format, each module contains:
*   `engine.py`: Core mathematical logic and algorithm implementation.
*   `analysis.py`: Script for data generation, experimentation, and visualization.
*   `results/`: Exported graphical analysis and performance metrics.

---
**Author:** [Your Name] – Bachelor of Artificial Intelligence
