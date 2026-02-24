import numpy as np
from scipy.linalg import khatri_rao
from scipy.io import loadmat
import matplotlib.pyplot as plt

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
def estimate_katri_E_full(ch_meas, conf_matr, lam=1e-5):
    CtC = conf_matr.T @ conf_matr
    reg = lam * np.eye(CtC.shape[0])
    inv_reg = np.linalg.solve(CtC + reg, conf_matr.T)
    E_hat = ch_meas @ inv_reg
    return E_hat

def estimate_katri_E_full_bias(ch_meas, conf_matr, lam=1e-5):
    """
    Ridge regression:
    ch_meas ≈ E @ conf_matr + b
    """

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

# Stack a and 1 for the intercept
def estim_lsq(a,h):
    A = np.vstack([a, np.ones_like(a)]).T

    # Least squares solution: [x, b]
    estim, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
    h_tilde, b_est = estim

    #print("Estimated x:", h_tilde)
    #print("Estimated b:", b_est)
    return h_tilde, b_est
def estim_coeff(vec_ones,theta):
    coeff = []
    for el in vec_ones:
        num_minus = 4-el
        coeff.append(el+ num_minus*np.exp(1j*theta))
    return coeff


def plot_mod_phase(channel):
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Magnitude plot
    ax1.plot(np.abs(channel), marker='o')
    ax1.set_title('Magnitude of Channel Coefficients')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True)

    # Phase plot
    ax2.plot(np.angle(channel, deg=True), marker='o', color='orange')
    ax2.set_title('Phase of Channel Coefficients')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Phase (degrees)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

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