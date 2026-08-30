import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from engine import ICADecompositionEngine
import os

# Setup results directory
if not os.path.exists('results'): os.makedirs('results')

def scale_img(s):
    return (s - s.min()) / (s.max() - s.min())

def show_image(img, ax):
    # Reshape to (Channels, Width, Height) -> (3, 32, 32)
    # Transpose to (32, 32, 3) for imshow
    ax.imshow(img.reshape(3, 32, 32).transpose((1, 2, 0)))
    ax.axis('off')

def analyze_cifar():
    print("--- Task 4: CIFAR-10 Decomposition ---")
    # Load dataset (Ensuring data exists)
    try:
        X = np.load('cifar-10.npy').astype(np.float64)
    except:
        print("cifar-10.npy not found. Skipping...")
        return

    pca_comp, ica_comp = ICADecompositionEngine.compare_decompositions(X)

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    plt.suptitle("Feature Extraction: PCA (Holistic) vs ICA (Localized/Sparse)", fontsize=14)

    for i in range(10):
        show_image(scale_img(pca_comp[i]), axes[0, i])
        if i == 0: axes[0, i].set_ylabel("PCA", rotation=0, labelpad=30, fontsize=12)
        
        show_image(scale_img(ica_comp[i]), axes[1, i])
        if i == 0: axes[1, i].set_ylabel("ICA", rotation=0, labelpad=30, fontsize=12)

    plt.tight_layout()
    plt.savefig('results/cifar_pca_vs_ica.png')
    plt.show()

def analyze_audio():
    print("\n--- Task 5: Cocktail Party Problem ---")
    try:
        # Load audio files
        _, b = wav.read('birds.wav')
        _, o = wav.read('ocean.wav')
        _, f = wav.read('frogs.wav')
    except:
        print("Audio files missing. Skipping...")
        return

    # Truncate to minimum length
    length = min(len(b), len(o), len(f))
    S = np.stack([b[:length, 0], o[:length, 0], f[:length, 0]], axis=1).astype(np.float32)

    # Mix
    A = np.array([[0.5, 0.5, 0.2], [0.3, 0.2, 0.8], [0.1, 0.7, 0.4]])
    X = ICADecompositionEngine.mix_signals(S, A)

    # Recover
    S_rec = ICADecompositionEngine.unmix_signals(X)
    S_rec_scaled = ICADecompositionEngine.normalize_signals(S_rec)

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    titles = ['Original (Birds)', 'Mixed Signal (Microphone 1)', 'Recovered (ICA Source 1)']
    
    axes[0].plot(S[:2000, 0], color='blue', alpha=0.7)
    axes[1].plot(X[:2000, 0], color='red', alpha=0.7)
    axes[2].plot(S_rec_scaled[:2000, 0], color='green', alpha=0.7)
    
    for i, title in enumerate(titles):
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('results/cocktail_party_recovery.png')
    plt.show()

if __name__ == "__main__":
    analyze_cifar()
    analyze_audio()
