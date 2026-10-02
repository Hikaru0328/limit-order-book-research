"""Unbounded unit queue with arrivals, market consumption and cancellation."""

from dataclasses import dataclass
import math

import numpy as np

from queue_models.queue_depletion import QueueSample


@dataclass(frozen=True)
class Config:
    initial: int = 3
    arrival: float = 2.0
    removal: float = 1.0
    theta: float = 0.2

    def __post_init__(self):
        if type(self.initial) is not int or self.initial < 1:
            raise ValueError("initial must be a positive integer")
        if not all(math.isfinite(x) and x >= 0 for x in (self.arrival, self.removal, self.theta)):
            raise ValueError("rates must be finite and nonnegative")
        if self.theta <= 0:
            raise ValueError("this model requires positive per-order cancellation")


def mean_reference(config, quantity=None, tolerance=1e-10, max_terms=100000):
    """Positive series with a geometric bound on omitted terms, excluding roundoff."""
    q = config.initial if quantity is None else quantity
    if type(q) is not int or q < 0 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid quantity or tolerance")
    if type(max_terms) is not int or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    value, bound, terms = 0., 0., 0
    for k in range(1, q + 1):
        term = 1 / (config.removal + config.theta * k)
        total = 0.
        for n in range(max_terms):
            total += term
            ratio = config.arrival / (config.removal + config.theta * (k + n + 1))
            tail = term * ratio / (1 - ratio) if ratio < 1 else math.inf
            if tail <= tolerance / max(q, 1):
                break
            term *= ratio
            if not math.isfinite(term + total):
                raise OverflowError("series exceeds floating point range")
        else:
            raise RuntimeError("series did not reach requested tolerance")
        value += total
        bound += tail
        terms += n + 1
    return dict(mean=value, tail_bound=bound, terms=terms)


def boundary_probability(config, upper):
    if type(upper) is not int or upper <= config.initial:
        raise ValueError("upper must exceed initial")
    if config.arrival == 0:
        return 0.
    log_w = np.zeros(upper)
    k = np.arange(1, upper)
    log_w[1:] = np.cumsum(np.log((config.removal + config.theta*k) / config.arrival))
    def logsum(x):
        m = x.max()
        return m + np.log(np.exp(x-m).sum())
    return float(np.exp(logsum(log_w[:config.initial]) - logsum(log_w)))


def finite_exit_mean(config, upper):
    """Independent finite tridiagonal boundary-value solve, absorbing at 0 and M."""
    boundary_probability(config, upper)
    k = np.arange(1, upper)
    death = config.removal + config.theta*k
    matrix = np.diag(config.arrival + death)
    matrix += np.diag(np.full(upper-2, -config.arrival), 1)
    matrix += np.diag(-death[1:], -1)
    return float(np.linalg.solve(matrix, np.ones(upper-1))[config.initial-1])


def simulate(config, cutoffs, samples=3000, seed=0):
    """Exact exponential events with snapshots; no upper queue boundary.

    Resampling clocks at administrative cutoffs is valid by memorylessness.
    Absorbed paths retain their hit time. Snapshot.remaining is the state at
    that cutoff and must not be reused as the state at an earlier cutoff.
    """
    cuts = np.asarray(cutoffs, dtype=float)
    if (cuts.ndim != 1 or not len(cuts) or not np.isfinite(cuts).all()
            or np.any(cuts < 0) or np.any(np.diff(cuts) <= 0)):
        raise ValueError("cutoffs must be finite, nonnegative and strictly increasing")
    if type(samples) is not int or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    rng = np.random.default_rng(seed)
    quantity = np.full(samples, config.initial, dtype=np.int64)
    time = np.zeros(samples)
    hit_time = np.full(samples, np.inf)
    snapshots = []
    for cutoff in cuts:
        active = np.flatnonzero(quantity > 0)
        while len(active):
            rate = config.arrival + config.removal + config.theta*quantity[active]
            event = time[active] + rng.exponential(1/rate)
            before = event <= cutoff
            time[active[~before]] = cutoff
            active, rate = active[before], rate[before]
            time[active] = event[before]
            quantity[active] += np.where(rng.random(len(active)) < config.arrival/rate, 1, -1)
            depleted = quantity[active] == 0
            hit_time[active[depleted]] = time[active[depleted]]
            active = active[~depleted]
        snapshots.append(QueueSample(np.minimum(hit_time, cutoff), np.isfinite(hit_time),
                                     quantity.copy(), float(cutoff)))
    return snapshots


def corrected_mean(config, sample):
    """Model-based residual-time correction; not an empirical uncensored sample."""
    states = np.unique(sample.remaining)
    refs = {int(k): mean_reference(config, int(k)) for k in states}
    residual = np.array([refs[int(k)]["mean"] for k in sample.remaining])
    values = sample.duration + residual
    return dict(corrected_mean=float(values.mean()),
                corrected_se=float(values.std(ddof=1)/np.sqrt(len(values))),
                correction_tail_bound=max(r["tail_bound"] for r in refs.values()))


def finite_reference(config, cutoffs, upper=64):
    """Uniformized killed chain, with upper-boundary and Poisson error budgets.

    Survival refers to exit from (0,M), hence is a lower bound for survival of
    the original queue. Add p(hit M before 0) and Poisson tail for an upper
    bound. RMST errors are at most cutoff times those probability bounds.
    """
    p_upper = boundary_probability(config, upper)
    cuts = np.asarray(cutoffs, dtype=float)
    if cuts.ndim != 1 or not len(cuts) or not np.isfinite(cuts).all() or np.any(cuts < 0):
        raise ValueError("invalid cutoffs")
    k = np.arange(1, upper)
    death = config.removal + config.theta*k
    rate = float(config.arrival + death[-1])
    max_mean = rate * cuts.max()
    steps = int(math.ceil(max_mean + 12*np.sqrt(max_mean+1) + 100))
    probability = np.zeros(upper-1)
    probability[config.initial-1] = 1.
    survival_n = np.empty(steps+1)
    survival_n[0] = 1.
    stay = 1 - (config.arrival+death)/rate
    for n in range(1, steps+1):
        following = probability * stay
        following[1:] += probability[:-1] * (config.arrival/rate)
        following[:-1] += probability[1:] * (death[1:]/rate)
        probability = following
        survival_n[n] = probability.sum()
    n = np.arange(steps+1)
    log_factorial = np.zeros(steps+1)
    log_factorial[1:] = np.cumsum(np.log(n[1:]))
    rows = []
    for cutoff in cuts:
        mean = rate*cutoff
        if mean == 0:
            pmf = np.zeros(steps+1)
            pmf[0] = 1.
            tail = 0.
        else:
            pmf = np.exp(-mean + n*np.log(mean) - log_factorial)
            tail = math.exp(min(0., -mean + steps*(1+math.log(mean/steps))))
        after = np.zeros(steps+1)
        after[:-1] = np.cumsum(pmf[:0:-1])[::-1]
        survival = float(pmf @ survival_n)
        rmst = float(after @ survival_n / rate)
        rows.append(dict(cutoff=float(cutoff), survival=survival, restricted_mean=rmst,
                         survival_error=p_upper+tail, rmst_error=float(cutoff*(p_upper+tail)),
                         upper_hit_bound=p_upper, poisson_tail_bound=tail, upper=upper))
    return rows
