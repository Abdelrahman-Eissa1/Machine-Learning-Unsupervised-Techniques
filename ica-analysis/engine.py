import numpy as np
from sklearn.decomposition import PCA, FastICA

class ICADecompositionEngine:
    """Engine for ICA vs PCA comparison and Blind Source Separation."""
    
    @staticmethod
    def compare_decompositions(X, n_components=100):
        """Runs both PCA and ICA on a dataset."""
        pca = PCA(n_components=n_components, random_state=42)
        ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
        
        pca_comps = pca.fit(X).components_
        ica_comps = ica.fit(X).components_
        
        return pca_comps, ica_comps

    @staticmethod
    def mix_signals(S, A):
        """Simulates the mixing of source signals."""
        return S @ A.T

    @staticmethod
    def unmix_signals(X, n_sources=3):
        """Recovers original sources from mixed signals using FastICA."""
        ica = FastICA(n_components=n_sources, random_state=42)
        S_recovered = ica.fit_transform(X)
        return S_recovered

    @staticmethod
    def normalize_signals(S):
        """Scales signals to [-1, 1] range for audio processing."""
        def norm(s):
            return 2 * (s - s.min()) / (s.max() - s.min()) - 1
        return np.array([norm(S[:, i]) for i in range(S.shape[1])]).T