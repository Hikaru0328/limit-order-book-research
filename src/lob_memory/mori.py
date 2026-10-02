"""Linear Mori projection from stationary correlations, separate from VAR."""
import numpy as np


def covariance_sequence(y, max_lag, mean=None):
    y = np.asarray(y, float)
    if y.ndim != 2 or not 0 <= max_lag < len(y):
        raise ValueError('require a 2D series and 0 <= max_lag < sample count')
    z = y - (y.mean(axis=0) if mean is None else mean)
    return np.array([z[k:].T @ z[:len(z)-k] / (len(z)-k)
                     for k in range(max_lag+1)])


def from_covariances(c):
    """C[0..L+1] -> Omega[0..L]; no ridge or joint history regression."""
    c = np.asarray(c, float)
    if c.ndim != 3 or c.shape[1] != c.shape[2] or len(c) < 2:
        raise ValueError('need square C[0..L+1] matrices')
    if not np.all(np.isfinite(c)) or np.linalg.cond(c[0]) > 1e12:
        raise ValueError('nonfinite or ill-conditioned covariance; remove constant/redundant variables')
    omega = []
    for n in range(len(c)-1):
        residual = c[n+1].copy()
        for k in range(n):
            residual -= omega[k] @ c[n-k]
        omega.append(np.linalg.solve(c[0].T, residual.T).T)
    return np.array(omega)


def whiten_kernel(omega, covariance):
    """Similarity transform using a positive definite training covariance."""
    values, vectors = np.linalg.eigh(covariance)
    if values.min() <= 0:
        raise ValueError('covariance must be positive definite')
    root = (vectors * np.sqrt(values)) @ vectors.T
    inverse = (vectors / np.sqrt(values)) @ vectors.T
    return np.array([inverse @ k @ root for k in omega])


def origin_residuals(y, omega):
    """Finite-origin exact-convention residuals W_n, not innovations.

    y has shape (independent origins, L+2, dimension); returns W_0..W_L.
    Orthogonality is against y[:,0], not against every past observation.
    """
    y = np.asarray(y, float)
    if y.ndim != 3 or y.shape[1] < len(omega)+1:
        raise ValueError('need independent trajectory segments of length L+2')
    residuals = []
    for n in range(len(omega)):
        w = y[:, n+1].copy()
        for k in range(n+1):
            w -= y[:, n-k] @ omega[k].T
        residuals.append(w)
    return np.stack(residuals, axis=1)
