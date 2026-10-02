"""Unit-order pure-death queue with cancellation and market consumption."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class DepletionConfig:
    initial: int = 100
    theta: float = 0.2
    mu: float = 2.0

    def __post_init__(self):
        if isinstance(self.initial, bool) or not isinstance(self.initial, Integral) or self.initial < 1:
            raise ValueError("initial must be a positive integer")
        if any(not np.isfinite(x) or x < 0 for x in (self.theta, self.mu)):
            raise ValueError("rates must be finite and nonnegative")
        if self.theta + self.mu == 0:
            raise ValueError("at least one removal mechanism must be active")


def stage_theory(config):
    """Arrays indexed in ascending quantity order: column 0 is the last order."""
    quantity = np.arange(1, config.initial + 1)
    rate = config.mu + config.theta * quantity
    mean = 1 / rate
    variance = mean**2
    return {"quantity": quantity, "rate": rate, "mean": mean, "variance": variance,
            "mean_share": mean / mean.sum(), "variance_share": variance / variance.sum(),
            "market_probability": config.mu / rate}


def theory(config):
    stages = stage_theory(config)
    mean, variance = stages["mean"].sum(), stages["variance"].sum()
    return {"mean": float(mean), "variance": float(variance),
            "cv": float(np.sqrt(variance) / mean),
            "fourth_central": float(6 * np.sum(stages["mean"]**4) + 3 * variance**2)}


def simulate(config, samples=20000, seed=0, method="combined"):
    """Return per-path, per-quantity waits; all paths reach zero without censoring.

    Combined draws Exp(mu + theta*k). Competing draws two independent clocks
    at each stage and takes their minimum. Memorylessness permits fresh clocks
    after every unit removal. No arrivals, resets, or price changes are modeled.
    """
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    if method not in ("combined", "competing"):
        raise ValueError("unknown method")
    rng = np.random.default_rng(seed)
    k = np.arange(1, config.initial + 1)
    shape = (samples, config.initial)
    if method == "combined":
        return rng.exponential(1 / (config.mu + config.theta * k), shape)
    cancel = (rng.exponential(1 / (config.theta * k), shape)
              if config.theta > 0 else np.full(shape, np.inf))
    market = (rng.exponential(1 / config.mu, shape)
              if config.mu > 0 else np.full(shape, np.inf))
    return np.minimum(cancel, market)


def assess(config, waits):
    waits = np.asarray(waits, dtype=float)
    if (waits.ndim != 2 or waits.shape[1] != config.initial or len(waits) < 2
            or not np.isfinite(waits).all() or (waits < 0).any()):
        raise ValueError("expected finite nonnegative per-path per-quantity waits")
    total = waits.sum(axis=1)
    n = len(total)
    ref = theory(config)
    mean_se = np.sqrt(ref["variance"] / n)
    var_se = np.sqrt((ref["fourth_central"]
                     - (n - 3) / (n - 1) * ref["variance"]**2) / n)
    mean, var = total.mean(), total.var(ddof=1)
    return {"q": config.initial, "theta": config.theta, "mu": config.mu, "n": n,
            "mean": float(mean), "theory_mean": ref["mean"],
            "variance": float(var), "theory_variance": ref["variance"],
            "cv": float(np.sqrt(var) / mean), "theory_cv": ref["cv"],
            "mean_se": float(mean_se), "variance_se": float(var_se),
            "mean_z": float((mean - ref["mean"]) / mean_se),
            "variance_z": float((var - ref["variance"]) / var_se)}
