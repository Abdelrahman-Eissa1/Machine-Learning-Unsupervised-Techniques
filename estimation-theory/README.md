## 🔢 Empirical Results
The implementation was validated using $N=500$ samples with a true mean $\mu=5.0$. The results demonstrate the inverse relationship between stochastic noise and estimator precision:

| Noise Level ($\sigma$) | Fisher Information | MLE Estimate ($\hat{\mu}$) | Precision |
| :--- | :--- | :--- | :--- |
| **Low Noise (1.0)** | 500.00 | 4.92 | **High** |
| **Medium Noise (3.0)** | 55.56 | 5.12 | **Moderate** |
| **High Noise (10.0)** | 5.00 | 4.56 | **Low** |

**Observation:** As $\sigma$ increases, the log-likelihood surface flattens, reducing the Fisher Information and increasing the variance of the maximum likelihood estimate, as governed by the Cramér-Rao Lower Bound.
