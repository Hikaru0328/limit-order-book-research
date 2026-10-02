"""Stationary linear-Gaussian benchmarks with independent population constructions."""
from __future__ import annotations
import numpy as np


def stationary_covariance(a, q):
    a, q = np.asarray(a, float), np.asarray(q, float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or q.shape != a.shape:
        raise ValueError('A and Q must be equally sized square matrices')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(q)):
        raise ValueError('matrices must be finite')
    if np.max(np.abs(np.linalg.eigvals(a))) >= 1:
        raise ValueError('A must be stable')
    if not np.allclose(q, q.T) or np.linalg.eigvalsh(q).min() < -1e-12:
        raise ValueError('Q must be symmetric positive semidefinite')
    n = len(a)
    s = np.linalg.solve(np.eye(n*n) - np.kron(a, a), q.ravel()).reshape(n, n)
    return (s + s.T) / 2


def simulate(a, q, n, seed=0):
    """Start in the exact stationary distribution; no burn-in is required."""
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError('n must be an integer >= 2')
    s = stationary_covariance(a, q)
    rng = np.random.default_rng(seed)
    x = np.empty((n, len(a)))
    x[0] = rng.multivariate_normal(np.zeros(len(a)), s)
    noise = rng.multivariate_normal(np.zeros(len(a)), q, n-1)
    for t in range(n-1):
        x[t+1] = a @ x[t] + noise[t]
    return x


def block_dynamics(a, q, b):
    """Exact endpoint transition and accumulated innovation covariance."""
    if not isinstance(b, (int, np.integer)) or b < 1:
        raise ValueError('b must be a positive integer')
    stationary_covariance(a, q)
    ab, qb = np.eye(len(a)), np.zeros_like(q, dtype=float)
    for _ in range(b):
        qb += ab @ q @ ab.T
        ab = a @ ab
    return ab, qb


def aggregate(x, b, mode='mean'):
    """Nonoverlapping complete blocks. Drop the final incomplete block."""
    x = np.asarray(x, float)
    if x.ndim != 2 or not isinstance(b, (int, np.integer)) or b < 1:
        raise ValueError('require a 2D series and positive integer block size')
    if len(x) < b:
        raise ValueError('not enough samples for one block')
    blocks = x[:len(x)//b*b].reshape(-1, b, x.shape[1])
    if mode == 'endpoint':
        return blocks[:, -1].copy()
    if mode == 'sum':
        return blocks.sum(axis=1)
    if mode == 'mean':
        return blocks.mean(axis=1)
    raise ValueError('mode must be endpoint, sum, or mean')


def population_covariances(a, q, c, b, max_lag, mode='endpoint'):
    """C[k]=E[Y_(j+k) Y_j^T], from sums of microscopic covariances."""
    block_dynamics(a, q, b)
    if max_lag < 0:
        raise ValueError('max_lag must be nonnegative')
    c = np.asarray(c, float)
    s = stationary_covariance(a, q)
    gamma = [s]
    for _ in range((max_lag+1)*b):
        gamma.append(a @ gamma[-1])
    if mode == 'endpoint':
        return np.array([c @ gamma[k*b] @ c.T for k in range(max_lag+1)])
    if mode not in ('sum', 'mean'):
        raise ValueError('unknown mode')
    out = []
    for k in range(max_lag+1):
        cov = np.zeros_like(s)
        for i in range(b):
            for j in range(b):
                delta = k*b+i-j
                cov += gamma[delta] if delta >= 0 else gamma[-delta].T
        out.append(c @ cov @ c.T / (b*b if mode == 'mean' else 1))
    return np.array(out)


def lifted_block_model(a, q, c, b, mode='mean'):
    """Markov state [observed block sum/mean, microscopic endpoint].

    The same event innovations drive both components. Their cross covariance
    must be retained. At b=1 this lift can be singular; use the endpoint model.
    """
    ab, _ = block_dynamics(a, q, b)
    if mode not in ('sum', 'mean'):
        raise ValueError('lift supports sum or mean')
    c = np.asarray(c, float)
    d, n = c.shape
    powers = [np.linalg.matrix_power(a, i) for i in range(b+1)]
    scale = b if mode == 'mean' else 1
    f = np.zeros((d+n, d+n))
    f[:d, d:] = c @ sum(powers[1:]) / scale
    f[d:, d:] = ab
    v = np.zeros_like(f)
    for r in range(1, b+1):
        g = np.vstack((c @ sum(powers[:b-r+1]) / scale, powers[b-r]))
        v += g @ q @ g.T
    return f, v


def exact_mori(a, q, observed, memory_lags):
    """Analytic linear Mori coefficients after covariance orthogonalization.

    The leading `observed` coordinates are resolved. Output[0] is the Markov
    matrix; output[k>=1]=B D**(k-1) C in orthogonalized coordinates.
    """
    s = stationary_covariance(a, q)
    n, d = len(a), observed
    if not 1 <= d <= n or memory_lags < 0:
        raise ValueError('invalid observed dimension or lag count')
    t = np.eye(n)
    t[d:, :d] = -np.linalg.solve(s[:d, :d], s[:d, d:]).T
    f = t @ a @ np.linalg.inv(t)
    out = [f[:d, :d]]
    power = np.eye(n-d)
    for _ in range(memory_lags):
        out.append(f[:d, d:] @ power @ f[d:, :d])
        power = power @ f[d:, d:]
    return np.array(out)
