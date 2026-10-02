"""Minimal RG diagnostics for scale-dependent matrix memory kernels.

Kernel convention
-----------------
kernel[k] is the matrix memory operator at block lag k+1, so the array has
shape (n_lags, n_variables, n_variables).

These utilities deliberately do not estimate a kernel. They summarize a
kernel that has already been estimated by a Mori projection procedure; VAR coefficients are separate predictors.
"""

from __future__ import annotations

import numpy as np


def lag_norms(kernel: np.ndarray, ord: str = "fro") -> np.ndarray:
    """Return ||K_k|| for each lag."""
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 3:
        raise ValueError("kernel must have shape (n_lags, d, d)")
    return np.asarray([np.linalg.norm(k, ord=ord) for k in kernel])


def memory_strength(kernel: np.ndarray, ord: str = "fro") -> float:
    """M_b = sum_{k>=1} ||K_k||."""
    return float(lag_norms(kernel, ord=ord).sum())


def characteristic_block_lag(kernel: np.ndarray, ord: str = "fro") -> float:
    """theta_b = sum k||K_k|| / sum ||K_k||, with lags starting at 1."""
    norms = lag_norms(kernel, ord=ord)
    total = norms.sum()
    if total == 0:
        return 0.0
    lags = np.arange(1, len(norms) + 1, dtype=float)
    return float(np.dot(lags, norms) / total)


def characteristic_event_horizon(kernel: np.ndarray, block_size: int, ord: str = "fro") -> float:
    """xi_b = b * theta_b, in microscopic event units."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return float(block_size * characteristic_block_lag(kernel, ord=ord))


def logscale_beta(scales: np.ndarray, couplings: np.ndarray) -> np.ndarray:
    """Finite-difference beta = dg/d ln b on an arbitrary increasing scale grid.

    Returns beta values at intervals; beta[i] corresponds to scales[i] -> scales[i+1].
    """
    scales = np.asarray(scales, dtype=float)
    couplings = np.asarray(couplings, dtype=float)
    if scales.ndim != 1 or couplings.shape[0] != scales.size:
        raise ValueError("first coupling axis must match one-dimensional scales")
    if np.any(scales <= 0) or np.any(np.diff(scales) <= 0):
        raise ValueError("scales must be strictly increasing and positive")
    dlog = np.diff(np.log(scales))
    reshape = (dlog.size,) + (1,) * (couplings.ndim - 1)
    return np.diff(couplings, axis=0) / dlog.reshape(reshape)
