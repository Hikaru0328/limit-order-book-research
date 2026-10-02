"""Exact sampling of the lecture's independent and alternating jump models.

Terminal samples use Poisson/binomial laws without time discretization. Paths
use exponential waiting times, retaining the actual event ordering.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ToyConfig:
    rate: float = 10.0
    p_up: float = 0.5
    delta: float = 1.0
    model: str = "independent"

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.rate, self.p_up, self.delta)):
            raise ValueError("Parameters must be finite")
        if self.rate < 0 or self.delta <= 0 or not 0 <= self.p_up <= 1:
            raise ValueError("Invalid rate, probability or jump size")
        if self.model not in ("independent", "alternating"):
            raise ValueError("Unknown model")
        if self.model == "alternating" and self.p_up != 0.5:
            raise ValueError("p_up is not a parameter of the alternating model; leave at 0.5")


def _horizon(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("Horizon must be finite and nonnegative")


def sample_terminal(config, horizon, samples, rng):
    """Return displacement, quadratic variation and count for independent paths.

    Alternation starts at price zero and the first jump is always upward.
    """
    _horizon(horizon)
    if type(samples) is not int or samples < 1:
        raise ValueError("Positive integer sample count required")
    count = rng.poisson(config.rate * horizon, samples)
    if config.model == "independent":
        up = rng.binomial(count, config.p_up)
        displacement = config.delta * (2 * up - count)
    else:
        displacement = config.delta * (count % 2)
    return displacement, config.delta**2 * count, count


def sample_fixed_count(config, count, samples, rng):
    """Alternating case randomizes its first direction for matched marginals."""
    if type(count) is not int or count < 0 or type(samples) is not int or samples < 1:
        raise ValueError("Integer count >= 0 and samples > 0 required")
    if config.model == "independent":
        up = rng.binomial(count, config.p_up, samples)
        return config.delta * (2 * up - count)
    first_sign = rng.choice([-1, 1], samples)
    return config.delta * (count % 2) * first_sign


def event_path(config, horizon, rng):
    """Step-function path including time zero and the terminal observation."""
    _horizon(horizon)
    times, prices, qv = [0.0], [0.0], [0.0]
    time, price, count = 0.0, 0.0, 0
    if config.rate:
        while True:
            time += rng.exponential(1 / config.rate)
            if time > horizon:
                break
            if config.model == "independent":
                sign = 1 if rng.random() < config.p_up else -1
            else:
                sign = 1 if count % 2 == 0 else -1
            price += sign * config.delta
            count += 1
            times.append(time)
            prices.append(price)
            qv.append(count * config.delta**2)
    if horizon > 0:
        times.append(horizon)
        prices.append(price)
        qv.append(count * config.delta**2)
    return np.array(times), np.array(prices), np.array(qv)


def theory(config, horizon):
    _horizon(horizon)
    activity = config.rate * horizon
    if config.model == "independent":
        mean = config.delta * (2 * config.p_up - 1) * activity
        variance = config.delta**2 * activity
        diffusion = config.delta**2 * config.rate
    else:
        odd_probability = -math.expm1(-2 * activity) / 2
        mean = config.delta * odd_probability
        variance = config.delta**2 * (-math.expm1(-4 * activity)) / 4
        diffusion = 0.0
    return dict(mean=mean, variance=variance, second_moment=variance + mean**2,
                quadratic_variation=config.delta**2 * activity,
                count=activity, long_run_variance_rate=diffusion)


def estimate_batches(config, horizon, seed, batches=30, per_batch=2000):
    """Independent batches give MC standard errors, not time-series pseudo-samples."""
    if batches < 2 or per_batch < 2:
        raise ValueError("At least two batches and samples per batch required")
    rng = np.random.default_rng(seed)
    estimates = {k: [] for k in ("mean", "variance", "second_moment", "quadratic_variation", "count")}
    for _ in range(batches):
        x, qv, count = sample_terminal(config, horizon, per_batch, rng)
        for key, value in dict(mean=x.mean(), variance=x.var(ddof=1),
                               second_moment=np.mean(x*x), quadratic_variation=qv.mean(),
                               count=count.mean()).items():
            estimates[key].append(value)
    target = theory(config, horizon)
    rows = []
    for metric, values in estimates.items():
        estimate = float(np.mean(values))
        se = float(np.std(values, ddof=1) / np.sqrt(batches))
        error = estimate - target[metric]
        rows.append(dict(metric=metric, estimate=estimate, theory=target[metric],
                         mc_se=se, error=error, within_4se=abs(error) <= 4*se + 1e-10))
    return rows
