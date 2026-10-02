"""Regenerative two-queue model: independent Poisson consumption, unit orders."""

import math
import numpy as np


def validate(ask, bid, p, rate, delta=1.0):
    if any(not isinstance(x, (int, np.integer)) or x < 1 for x in (ask, bid)):
        raise ValueError("Queue sizes must be positive integers")
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if not np.isfinite(rate) or rate <= 0 or not np.isfinite(delta) or delta <= 0:
        raise ValueError("rate and delta must be finite and positive")


def event_law(ask, bid, p):
    """Exact joint law of direction D and event count N (no time sampling)."""
    validate(ask, bid, p, 1.0)
    rows = []
    for k in range(bid):
        rows.append((1, ask + k, math.comb(ask+k-1, k)*p**ask*(1-p)**k))
    for k in range(ask):
        rows.append((-1, bid + k, math.comb(bid+k-1, k)*(1-p)**bid*p**k))
    return np.array(rows, dtype=float)


def theory(ask=2, bid=1, p=0.5, rate=2.0, delta=1.0):
    validate(ask, bid, p, rate, delta)
    d, n, w = event_law(ask, bid, p).T
    ej = np.sum(w*d)*delta
    et = np.sum(w*n)/rate
    et2 = np.sum(w*n*(n+1))/rate**2
    vj = delta**2-ej**2
    vt = et2-et**2
    cov = np.sum(w*d*n)*delta/rate-ej*et
    drift = ej/et
    independent = (vj+drift**2*vt)/et
    return dict(p_up=float(np.sum(w[d > 0])), mean_t=float(et), var_t=float(vt),
                cov_jt=float(cov), drift=float(drift),
                variance_rate=float(independent-2*drift*cov/et),
                independent_rate=float(independent))


def sample_cycles(rng, size, ask=2, bid=1, p=0.5, rate=2.0, delta=1.0):
    """Simulate the race event-by-event, independently of the analytic law."""
    validate(ask, bid, p, rate, delta)
    a = np.full(size, ask, dtype=int)
    b = np.full(size, bid, dtype=int)
    times = np.zeros(size)
    counts = np.zeros(size, dtype=int)
    while True:
        active = np.flatnonzero((a > 0) & (b > 0))
        if not len(active):
            break
        times[active] += rng.exponential(1/rate, len(active))
        up = rng.random(len(active)) < p
        a[active[up]] -= 1
        b[active[~up]] -= 1
        counts[active] += 1
    jumps = np.where(a == 0, delta, -delta)
    return jumps, times, counts


def cycle_estimate(jumps, times):
    drift = jumps.sum()/times.sum()
    return dict(drift=float(drift),
                variance_rate=float(np.mean((jumps-drift*times)**2)/times.mean()),
                cov_jt=float(np.mean((jumps-jumps.mean())*(times-times.mean()))))


def price_paths(rng, paths, grid, independent=False, **config):
    """Prices on fixed times; no jump is counted until its cycle completes.

    Independent mode uses a second fresh race for each waiting time, preserving
    both marginals without finite-pool shuffling or cross-cycle dependence.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not len(grid) or np.any(~np.isfinite(grid)) or np.any(grid < 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be finite, nonnegative and strictly increasing")
    if not isinstance(paths, int) or paths < 1:
        raise ValueError("paths must be a positive integer")

    def draw(size):
        j, t, _ = sample_cycles(rng, size, **config)
        if independent:
            _, t, _ = sample_cycles(rng, size, **config)
        return j, t

    pending, next_time = draw(paths)
    price = np.zeros(paths)
    output = np.empty((paths, len(grid)))
    for column, time in enumerate(grid):
        while True:
            active = np.flatnonzero(next_time <= time)
            if not len(active):
                break
            price[active] += pending[active]
            j, t = draw(len(active))
            pending[active] = j
            next_time[active] += t
        output[:, column] = price
    return output
