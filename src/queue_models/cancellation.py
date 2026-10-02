"""Pure cancellation queue: no arrivals, market orders, resets, or censoring."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class CancellationConfig:
    initial: int = 3
    theta: float = 0.2

    def __post_init__(self):
        if isinstance(self.initial, bool) or not isinstance(self.initial, Integral) or self.initial < 1:
            raise ValueError("initial must be a positive integer")
        if not np.isfinite(self.theta) or self.theta <= 0:
            raise ValueError("theta must be finite and positive")


def theory(config):
    k = np.arange(1, config.initial + 1, dtype=float)
    mean = np.sum(1 / k) / config.theta
    variance = np.sum(1 / k**2) / config.theta**2
    # Cumulants add for independent exponential stage durations.
    fourth_central = 6 * np.sum(1 / k**4) / config.theta**4 + 3 * variance**2
    return {"mean": float(mean), "variance": float(variance),
            "fourth_central": float(fourth_central),
            "cv": float(np.sqrt(variance) / mean)}


def cdf(config, time):
    t = np.asarray(time, dtype=float)
    if np.any(np.isnan(t)):
        raise ValueError("time must not contain NaN")
    # expm1 avoids cancellation near time zero.
    return (-np.expm1(-config.theta * np.maximum(t, 0))) ** config.initial


def simulate(config, samples=20000, seed=0, method="maximum", batch_size=256):
    for name, value in (("samples", samples), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if method not in ("maximum", "stages"):
        raise ValueError("method must be maximum or stages")
    rng = np.random.default_rng(seed)
    result = np.empty(samples)
    scales = 1 / (config.theta * np.arange(config.initial, 0, -1, dtype=float))
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        if method == "maximum":
            clocks = rng.exponential(1 / config.theta, (size, config.initial))
            result[start:start + size] = clocks.max(axis=1)
        else:
            durations = rng.exponential(scales, (size, config.initial))
            result[start:start + size] = durations.sum(axis=1)
    return result


def assess(config, sample, cdf_alpha=0.01):
    x = np.asarray(sample, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("sample must contain at least two finite nonnegative durations")
    if not 0 < cdf_alpha < 1:
        raise ValueError("cdf_alpha must be between zero and one")
    ref = theory(config)
    n = len(x)
    mean_se = np.sqrt(ref["variance"] / n)
    variance_se = np.sqrt((ref["fourth_central"]
                           - (n - 3) / (n - 1) * ref["variance"]**2) / n)
    f = cdf(config, np.sort(x))
    distance = max(np.max(np.arange(1, n + 1) / n - f),
                   np.max(f - np.arange(n) / n))
    bound = np.sqrt(np.log(2 / cdf_alpha) / (2 * n))
    mean, variance = float(x.mean()), float(x.var(ddof=1))
    return {"q": config.initial, "theta": config.theta, "n": n,
            "mean": mean, "theory_mean": ref["mean"], "mean_se": float(mean_se),
            "mean_z": float((mean - ref["mean"]) / mean_se),
            "variance": variance, "theory_variance": ref["variance"],
            "variance_se": float(variance_se),
            "variance_z": float((variance - ref["variance"]) / variance_se),
            "cv": float(np.sqrt(variance) / mean), "theory_cv": ref["cv"],
            "cdf_distance": float(distance), "cdf_bound": float(bound),
            "cdf_pass": bool(distance <= bound)}
