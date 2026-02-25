import numpy as np
from scipy.linalg import khatri_rao
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Function to convert hex string to matrix representing the RIS configuration
def hex_to_matrix(hex_string, N=16):
    # Remove the '!0X' prefix if present
    if hex_string.startswith('!0X'):
        hex_string = hex_string[3:]
    # Convert hex to binary string
    bin_str = ''.join(f"{int(char, 16):04b}" for char in hex_string)
    # Take only the first N*N bits (in case there are extra bits)
    bin_str = bin_str[:N*N]
    # Convert to numpy array and reshape
    arr = np.array(list(bin_str), dtype=int).reshape(N, N)
    return arr
# Function that estimates the Khatri-Rao product via Ridge Regression (LM)
def estimate_katri_E_full(ch_meas, conf_matr, lam=1e-5):
    CtC = conf_matr.T @ conf_matr
    reg = lam * np.eye(CtC.shape[0])
    inv_reg = np.linalg.solve(CtC + reg, conf_matr.T)
    E_hat = ch_meas @ inv_reg
    return E_hat

# Function that estimates the Khatri-Rao product with bias (with regularizer to prevent inversion explosion)
def estimate_katri_E_full_bias(ch_meas, conf_matr, lam=1e-5):
    # Augment with bias
    ones = np.ones((1, conf_matr.shape[1]))
    X = np.vstack([conf_matr, ones])   # (N_feat+1, N_samples)

    # Ridge in feature space
    XXt = X @ X.T
    reg = lam * np.eye(XXt.shape[0])
    W = ch_meas @ X.T @ np.linalg.inv(XXt + reg)

    E_hat = W[:, :-1]
    b_hat = W[:, -1]

    return E_hat, b_hat
# Function to display the RIS configuration matrix as an image
def display_matrix(matrix, idx,ax=None):
    if ax is None:
        plt.figure(figsize=(4, 4))
        ax = plt.gca()
    ax.imshow(matrix, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks([])  # optional: hide major ticks
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title(idx)