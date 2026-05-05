import numpy as np
from scipy.special import factorial

def poisson_log_likelihood(data, lam):
    """
    Computes the log-likelihood for a Poisson distribution.
    Formula: sum(x_i * ln(lambda) - lambda - ln(x_i!))
    """
    if lam <= 0:
        return -np.inf
    
    # Using log of factorial for numerical stability
    term1 = data * np.log(lam)
    term2 = lam
    term3 = np.log(factorial(data))
    
    return np.sum(term1 - term2 - term3)

def get_analytical_mle(data):
    """
    The analytical MLE for Poisson is the sample mean.
    Derivation: d/d_lambda [log_likelihood] = 0 => lambda_hat = mean(x)
    """
    return np.mean(data)

def get_numerical_mle(data, lam_set):
    """
    Finds the MLE by searching through a grid of possible lambda values.
    """
    log_likelihoods = [poisson_log_likelihood(data, l) for l in lam_set]
    idx_max = np.argmax(log_likelihoods)
    return lam_set[idx_max], log_likelihoods
