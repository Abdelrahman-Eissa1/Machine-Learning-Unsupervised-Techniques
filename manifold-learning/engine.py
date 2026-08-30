import numpy as np

class ManifoldEngine:
    """From-scratch implementation of SNE and t-SNE optimization."""

    @staticmethod
    def neg_squared_euc_dists(X):
        """Computes pairwise negative squared Euclidean distances."""
        sum_X = np.sum(np.square(X), axis=1)
        D = sum_X.reshape([-1, 1]) + sum_X.reshape([1, -1]) - 2 * np.dot(X, X.T)
        return -D

    @staticmethod
    def softmax(X):
        """Numerically stable softmax for distance matrices."""
        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
        np.fill_diagonal(e_x, 0.)
        e_x = e_x + 1e-8 
        return e_x / e_x.sum(axis=1).reshape([-1, 1])

    @staticmethod
    def calc_prob_matrix(distances, sigmas):
        """Calculates conditional probabilities P_j|i."""
        two_sig_sq = 2. * np.square(sigmas.reshape([-1, 1]))
        return ManifoldEngine.softmax(distances / two_sig_sq)

    @staticmethod
    def p_joint(X, target_perplexity):
        """Computes the symmetrized joint probability matrix P."""
        n = X.shape[0]
        distances = ManifoldEngine.neg_squared_euc_dists(X)
        
        # Helper for binary search
        def get_perp(sigma, dist_row):
            p_row = ManifoldEngine.softmax(dist_row / (2. * sigma**2))
            entropy = -np.sum(p_row * np.log2(p_row))
            return 2**entropy

        sigmas = []
        for i in range(n):
            # Binary search for sigma_i to match target perplexity
            sigmas.append(ManifoldEngine.binary_search(
                lambda s: get_perp(s, distances[i:i+1, :]), target_perplexity))
        
        P_cond = ManifoldEngine.calc_prob_matrix(distances, np.array(sigmas))
        return (P_cond + P_cond.T) / (2.0 * n)

    @staticmethod
    def binary_search(eval_fn, target, tol=1e-10, max_iter=100):
        low, high = 1e-20, 1000.
        for _ in range(max_iter):
            mid = (low + high) / 2.
            if eval_fn(mid) > target: high = mid
            else: low = mid
            if np.abs(eval_fn(mid) - target) <= tol: break
        return mid

    @staticmethod
    def tsne_grad(P, Y):
        """Computes the gradient for t-SNE using the Student t-distribution."""
        dists = -ManifoldEngine.neg_squared_euc_dists(Y)
        inv_dists = 1. / (1. + dists)
        np.fill_diagonal(inv_dists, 0.)
        Q = inv_dists / np.sum(inv_dists)
        
        # Gradient implementation
        pq_diff = (P - Q) * inv_dists
        grad = np.zeros_like(Y)
        for i in range(len(Y)):
            grad[i] = 4 * np.dot(pq_diff[i], Y[i] - Y)
        return grad, Q

    @staticmethod
    def estimate_tsne(X, P, iters=500, lr=10):
        """Optimization loop for t-SNE."""
        Y = np.random.RandomState(1).normal(0, 0.0001, [X.shape[0], 2])
        for i in range(iters):
            grad, _ = ManifoldEngine.tsne_grad(P, Y)
            Y -= lr * grad
        return Y