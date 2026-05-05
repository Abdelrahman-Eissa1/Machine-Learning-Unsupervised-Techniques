import numpy as np

def gauss_pdf(x, mu, sigma):
    """Computes the 1D Gaussian Probability Density Function."""
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu)**2) / (2 * (sigma**2)))

def log_likelihood(data, mu, sigma):
    """Calculates the Log-Likelihood for numerical stability."""
    return np.sum(np.log(gauss_pdf(data, mu, sigma)))

def calculate_log_likelihood_over_mu(data, sigma, mu_range):
    """Computes log-likelihood values over a range of possible mu values."""
    lnL_list = [log_likelihood(data, mu, sigma) for mu in mu_range]
    return np.array(lnL_list)

def fisher_information_mu(n, sigma):
    """Calculates the Fisher Information for the mean of a Gaussian."""
    return n / (sigma**2)

def calculate_crlb(n, sigma):
    """Calculates the Cramer-Rao Lower Bound."""
    return (sigma**2) / n
