"""Single best queue, stopped at depletion, with administrative censoring.

Unit limit arrivals have rate arrival; unit market removals have rate removal.
The queue is unbounded above. Zero is absorbing: no reset, cancellation or price
process is added. The deterministic reference uses Poisson uniformization.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class QueueConfig:
    initial: int = 3
    arrival: float = 1.0
    removal: float = 2.0

    def __post_init__(self):
        if type(self.initial) is not int or self.initial < 1:
            raise ValueError("Initial queue must be a positive integer")
        if not all(math.isfinite(x) and x >= 0 for x in (self.arrival, self.removal)):
            raise ValueError("Rates must be finite and nonnegative")

    @property
    def mean_depletion_time(self):
        if self.removal <= self.arrival:
            return math.inf
        return self.initial / (self.removal - self.arrival)

    @property
    def eventual_depletion_probability(self):
        if self.removal == 0:
            return 0.0
        if self.arrival <= self.removal:
            return 1.0
        return (self.removal / self.arrival)**self.initial


def _check_horizon(horizon):
    if not math.isfinite(horizon) or horizon < 0:
        raise ValueError("Horizon must be finite and nonnegative")


@dataclass
class QueueSample:
    duration: np.ndarray
    observed: np.ndarray
    remaining: np.ndarray
    horizon: float


def simulate(config, horizon, samples, seed):
    """Exact exponential clocks; unobserved depletion times are not fabricated."""
    _check_horizon(horizon)
    if type(samples) is not int or samples < 2:
        raise ValueError("At least two independent paths required")
    rng = np.random.default_rng(seed)
    times = np.zeros(samples)
    quantities = np.full(samples, config.initial, dtype=np.int64)
    duration = np.full(samples, horizon, dtype=float)
    observed = np.zeros(samples, dtype=bool)
    rate = config.arrival + config.removal
    if rate and horizon:
        active = np.arange(samples)
        while len(active):
            next_times = times[active] + rng.exponential(1 / rate, len(active))
            before_end = next_times <= horizon
            active = active[before_end]
            times[active] = next_times[before_end]
            quantities[active] += np.where(
                rng.random(len(active)) < config.arrival / rate, 1, -1)
            depleted = quantities[active] == 0
            done = active[depleted]
            duration[done] = times[done]
            observed[done] = True
            active = active[~depleted]
    return QueueSample(duration, observed, quantities, horizon)


def summarize(sample, cutoff):
    """Reuse the same paths at earlier cutoffs; nested estimates are correlated."""
    _check_horizon(cutoff)
    if cutoff > sample.horizon:
        raise ValueError("Cannot extrapolate beyond the simulated horizon")
    hit = sample.observed & (sample.duration <= cutoff)
    restricted = np.minimum(sample.duration, cutoff)
    n = len(restricted)
    survival = 1 - hit.mean()
    # Four-standard-error Wilson interval remains useful at zero censor count.
    z = 4.0
    center = (survival + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(survival*(1-survival)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return dict(cutoff=cutoff, samples=n, depleted=int(hit.sum()),
                censored=int((~hit).sum()), survival=survival,
                survival_low=max(0, center-half), survival_high=min(1, center+half),
                restricted_mean=float(restricted.mean()),
                restricted_se=float(restricted.std(ddof=1) / math.sqrt(n)),
                completed_only_mean=float(sample.duration[hit].mean()) if hit.any() else math.nan)


def reference(config, cutoffs, margin=12.0):
    """Deterministic survival and restricted means for an infinite-capacity queue.

    After at most M events no path can exceed initial+M. Recursion is exact on
    this reachable set, so there is no reflecting/truncating queue boundary.
    Only the Poisson event-count sum is truncated. Bounds exclude roundoff.
    """
    cutoffs = np.asarray(cutoffs, dtype=float)
    if cutoffs.ndim != 1 or not len(cutoffs):
        raise ValueError("A nonempty vector of cutoffs is required")
    for cutoff in cutoffs:
        _check_horizon(float(cutoff))
    if not math.isfinite(margin) or margin < 1:
        raise ValueError("Positive truncation margin required")
    rate = config.arrival + config.removal
    if rate == 0:
        return [dict(cutoff=float(t), survival=1., restricted_mean=float(t),
                     survival_tail_bound=0., rmst_tail_bound=0.) for t in cutoffs]
    max_mean = rate * max(cutoffs)
    steps = int(math.ceil(max_mean + margin * math.sqrt(max_mean + 1) + 100))
    probability = np.zeros(config.initial + steps + 2)
    probability[config.initial] = 1
    survival_n = np.empty(steps + 1)
    survival_n[0] = 1
    up = config.arrival / rate
    for step in range(1, steps + 1):
        following = np.zeros_like(probability)
        following[1:] += up * probability[:-1]
        following[:-1] += (1-up) * probability[1:]
        following[0] = 0  # Absorbed probability is never returned to the queue.
        probability = following
        survival_n[step] = probability.sum()
    n = np.arange(steps + 1)
    log_factorial = np.zeros(steps + 1)
    log_factorial[1:] = np.cumsum(np.log(n[1:]))
    rows = []
    for cutoff in cutoffs:
        mean = rate * cutoff
        if mean == 0:
            pmf = np.zeros(steps + 1)
            pmf[0] = 1
            bound = 0.0
        else:
            pmf = np.exp(-mean + n*np.log(mean) - log_factorial)
            # Chernoff bound for P(Poisson(mean) >= steps).
            bound = math.exp(min(0., -mean + steps*(1+math.log(mean/steps))))
        tail_after_n = np.zeros(steps + 1)
        tail_after_n[:-1] = np.cumsum(pmf[:0:-1])[::-1]
        rows.append(dict(cutoff=float(cutoff),
                         survival=float(np.dot(pmf, survival_n)),
                         restricted_mean=float(np.dot(tail_after_n, survival_n)/rate),
                         survival_tail_bound=bound, rmst_tail_bound=float(cutoff*bound)))
    return rows


def sample_path(config, horizon, seed):
    _check_horizon(horizon)
    rng = np.random.default_rng(seed)
    rate = config.arrival + config.removal
    times, sizes = [0.], [config.initial]
    while sizes[-1] and rate:
        time = times[-1] + rng.exponential(1/rate)
        if time > horizon:
            break
        times.append(time)
        sizes.append(sizes[-1] + (1 if rng.random() < config.arrival/rate else -1))
    if times[-1] < horizon:
        times.append(horizon)
        sizes.append(sizes[-1])
    return np.array(times), np.array(sizes)
