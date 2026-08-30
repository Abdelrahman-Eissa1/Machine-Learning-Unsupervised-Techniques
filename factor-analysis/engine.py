import numpy as np

class FactorAnalysisEngine:
    """
    Engine for Factor Analysis using Gradient Descent and Expectation Maximization.
    Demonstrates numerical stability via the Matrix Inversion Lemma.
    """
    
    @staticmethod
    def matrix_inverse_lemma(U, psi_vec):
        """
        Efficiently computes Q = (UU^T + Psi)^-1 using the Woodbury Identity.
        Complexity: O(m*l^2) instead of O(m^3).
        """
        m, l = U.shape
        inv_psi = np.diag(1.0 / psi_vec)
        inner_inv = np.linalg.inv(np.eye(l) + U.T @ inv_psi @ U)
        Q = inv_psi - (inv_psi @ U) @ inner_inv @ (U.T @ inv_psi)
        return Q

    @staticmethod
    def compute_loss(X, U, psi_vec):
        """Negative Log-Likelihood of the Factor Analysis model."""
        n, m = X.shape
        l = U.shape[1]
        C = (X.T @ X) / n
        
        # Log-det calculation using the matrix determinant lemma for stability
        log_det_psi = np.sum(np.log(psi_vec))
        inner = np.eye(l) + U.T @ np.diag(1.0/psi_vec) @ U
        log_det = log_det_psi + np.log(np.linalg.det(inner))
        
        Q = FactorAnalysisEngine.matrix_inverse_lemma(U, psi_vec)
        loss = (n / 2) * (log_det + np.trace(Q @ C))
        return loss

    @staticmethod
    def grad_step(X, U, psi_vec, lr):
        """One step of Gradient Descent using derived matrix gradients."""
        n, m = X.shape
        C = (X.T @ X) / n
        Q = FactorAnalysisEngine.matrix_inverse_lemma(U, psi_vec)
        
        # P is the error signal matrix: 0.5 * (Q - Q @ C @ Q)
        P = 0.5 * (Q - Q @ C @ Q)
        
        grad_U = 2 * P @ U
        grad_psi = np.diag(P)
        
        U_new = U - lr * grad_U
        psi_new = np.maximum(psi_vec - lr * grad_psi, 1e-6)
        return U_new, psi_new

    @staticmethod
    def em_step(X, U, psi_vec):
        """One step of Expectation Maximization with speedups."""
        n, m = X.shape
        l = U.shape[1]
        C = (X.T @ X) / n
        Q = FactorAnalysisEngine.matrix_inverse_lemma(U, psi_vec)
        
        # E-step: Sufficient statistics
        E_yx = C @ Q @ U
        E_yy = (np.eye(l) - U.T @ Q @ U) + U.T @ Q @ C @ Q @ U
        
        # M-step: Update parameters
        U_new = E_yx @ np.linalg.inv(E_yy)
        psi_new = np.diag(C - E_yx @ U_new.T)
        return U_new, np.maximum(psi_new, 1e-6)