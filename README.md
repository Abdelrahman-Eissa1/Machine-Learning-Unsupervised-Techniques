# Machine-Learning-Unsupervised-Techniques

A modular Python library of unsupervised machine learning implementations, covering statistical foundations, clustering theory, and dimensionality reduction from scratch using NumPy and Scikit-Learn. Based on the curriculum by Sepp Hochreiter.

## 🛠 Project Modules

### 📐 [1. Estimation Theory & Maximum Likelihood](./estimation-theory)
*   **Engineered** a modular statistical engine to compute Log-Likelihoods for Gaussian distributions.
*   **Quantified** estimator precision via Fisher Information and the Cramér-Rao Lower Bound (CRLB).

### 📈 [2. Poisson Likelihood Optimization](./poisson-likelihood-analysis)
*   **Derived** analytical Maximum Likelihood Estimators (MLE) for Poisson-distributed stochastic processes.
*   **Visualized** concave log-likelihood surfaces to identify global maxima.

### 🧬 [3. Principal Component Analysis](./principal-component-analysis)
*   **Deconstructed** high-dimensional datasets into orthogonal principal components using SVD.
*   **Demonstrated** feature importance ranking on multivariate datasets (Iris).

### 🖼️ [4. PCA Compression & Kernel Manifolds](./pca-kernel-compression)
*   **Eigenface Pipeline:** Built an image compression-reconstruction pipeline from scratch.
*   **The Kernel Trick:** Implemented **Kernel PCA** (RBF) to resolve non-linear manifolds.

### 🎤 [5. Independent Component Analysis (ICA)](./ica-analysis)
*   **Blind Source Separation:** Solved the **"Cocktail Party Problem"** by recovering independent audio sources from linear mixtures.
*   **Localized Feature Discovery:** Proved that ICA extracts edge-filters from images (CIFAR-10) while PCA extracts holistic structures.

## 🧰 Tech Stack
*   **Languages:** Python 3.x
*   **Scientific Computing:** NumPy, SciPy, Pandas
*   **Machine Learning:** Scikit-Learn (Metrics & Evaluation)
*   **Data Visualization:** Matplotlib, Seaborn

## 📂 Repository Structure
To maintain a professional library format, each module is organized as follows:
*   `engine.py`: Core mathematical logic, class wrappers, and algorithm implementation.
*   `analysis.py`: Script for data generation, experimentation, and visualization.
*   `results/`: Exported graphical analysis and performance metrics.

---
**Author:** Abdelrahman Eissa – Bachelor of Artificial Intelligence
