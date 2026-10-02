"""Known-parameter causal filters, including the absence of observed events.

No hidden flip times or hidden labels enter Observations or filter functions.
Price filtering propagates ALL queue/direction states; conditioning on a queue
snapshot at evaluation time must not be fed back into its price-only history.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class Observations:
    kind: str
    times: np.ndarray
    reward: np.ndarray
    queues: np.ndarray | None
    event: np.ndarray | None
    initial_queue: np.ndarray | None
    end: float


def observe(model, path, kind):
    if kind not in ("price", "queue_price", "marked"):
        raise ValueError("unknown observation kind")
    mask = path.reward != 0 if kind == "price" else path.event < 6
    return Observations(kind, path.times[mask].copy(), path.reward[mask].copy(),
                        None if kind == "price" else model.states[path.state[mask], :2].copy(),
                        path.event[mask].copy() if kind == "marked" else None,
                        None if kind == "price" else model.states[path.initial, :2].copy(), path.end)


def normalize(v):
    v = np.asarray(v, float)
    if not np.all(np.isfinite(v)) or v.min() < -1e-10:
        raise ArithmeticError("invalid filtering mass")
    v = np.maximum(v, 0.)
    mass = v.sum()
    if mass <= 0: raise ArithmeticError("observation has zero likelihood")
    return v / mass


class Survival:
    """Matrix-exponential action for a killed generator, without a time grid.

    Use diagonalization only if well conditioned; otherwise sparse exponential
    action. Both are finite-matrix calculations, not a discretized filter.
    The eigenvalue shift cancels at normalization and prevents long-gap underflow.
    """
    def __init__(self, generator):
        self.generator = np.asarray(generator, float)
        self.eigenvalues, self.vectors = eig(self.generator)
        self.condition = float(np.linalg.cond(self.vectors))
        self.spectral = self.condition < 1e6
        self.inverse = np.linalg.inv(self.vectors) if self.spectral else None
        self.shift = float(np.max(self.eigenvalues.real))
        self.shifted_transpose = csr_matrix((self.generator - self.shift * np.eye(len(generator))).T)
        self.fallbacks = 0

    def step(self, belief, dt):
        if not np.isfinite(dt) or dt < 0: raise ValueError("nonnegative dt required")
        if dt == 0: return belief.copy()
        if self.spectral:
            v = ((belief @ self.vectors) * np.exp((self.eigenvalues - self.shift) * dt)) @ self.inverse
            tolerance = 1e-11 * max(1., float(np.max(abs(v))))
            if np.max(abs(v.imag)) <= tolerance and v.real.min() >= -tolerance:
                return normalize(v.real)
        self.fallbacks += 1
        return normalize(expm_multiply(dt * self.shifted_transpose, belief))


def no_book_event(belief, hazards, kappa, dt):
    """Stable analytic exponential of the symmetric 2x2 killed generator."""
    if not np.isfinite(dt) or dt < 0: raise ValueError("nonnegative dt required")
    d = (hazards[1] - hazards[0]) / 2
    delta = np.hypot(d, kappa)
    decay = np.exp(-2 * delta * dt)
    c, v = (1 + decay) / 2, (1 - decay) / (2 * delta)
    a, b = belief
    return normalize(np.array([a * (c + v * d) + b * v * kappa,
                               a * v * kappa + b * (c - v * d)]))


def _validate_queries(observations, queries):
    queries = np.asarray(queries, float)
    if queries.ndim != 1 or not np.all(np.isfinite(queries)) or np.any(np.diff(queries) < 0):
        raise ValueError("sorted finite query times required")
    if np.any(queries < 0) or np.any(queries > observations.end):
        raise ValueError("query outside observation window")
    return queries


def queue_filter(model, observations, queries, initial_belief=None):
    if observations.kind not in ("queue_price", "marked"):
        raise ValueError("queue observations required")
    queries = _validate_queries(observations, queries)
    bid, ask = observations.initial_queue
    belief = model.current_queue_belief(bid, ask)
    if initial_belief is not None:
        if np.shape(initial_belief)!=(2,): raise ValueError("two initial probabilities required")
        belief=normalize(initial_belief)
    last = 0.; j = 0; result = []
    for t in queries:
        while j < len(observations.times) and observations.times[j] <= t:
            i = model.index(bid, ask, -1); indexes = np.array([i, i + 1])
            hazards = model.rates[indexes, :6].sum(1)
            belief = no_book_event(belief, hazards, model.parameters.kappa, observations.times[j] - last)
            next_queue = observations.queues[j]; reward = observations.reward[j]
            if observations.kind == "marked":
                e = observations.event[j]
                match = np.all(model.states[model.targets[indexes, e], :2] == next_queue, axis=1)
                match &= model.rewards[indexes, e] == reward
                likelihood = model.rates[indexes, e] * match
            else:
                targets = model.targets[indexes, :6]
                match = np.all(model.states[targets, :2] == next_queue, axis=2)
                match &= model.rewards[indexes, :6] == reward
                likelihood = np.sum(model.rates[indexes, :6] * match, axis=1)
            belief = normalize(belief * likelihood)
            bid, ask = next_queue; last = observations.times[j]; j += 1
        i = model.index(bid, ask, -1)
        belief = no_book_event(belief, model.rates[i:i + 2, :6].sum(1), model.parameters.kappa, t - last)
        last = t; result.append(belief.copy())
    return np.array(result).reshape(-1, 2)


def price_filter(model, observations, queries, propagator=None, initial_belief=None):
    if observations.kind != "price": raise ValueError("price observations required")
    queries = _validate_queries(observations, queries)
    if propagator is None:
        propagator = Survival(model.generator - model.price_jumps[1] - model.price_jumps[-1])
    belief = model.stationary.copy()
    if initial_belief is not None:
        if np.shape(initial_belief)!=(model.n,): raise ValueError("one initial probability per state required")
        belief=normalize(initial_belief)
    last = 0.; j = 0; result = []
    jump_transpose = {s: csr_matrix(model.price_jumps[s].T) for s in (-1, 1)}
    for t in queries:
        while j < len(observations.times) and observations.times[j] <= t:
            belief = propagator.step(belief, observations.times[j] - last)
            sign = 1 if observations.reward[j] > 0 else -1
            belief = normalize(jump_transpose[sign] @ belief)
            last = observations.times[j]; j += 1
        belief = propagator.step(belief, t - last)
        last = t; result.append(belief.copy())
    return np.array(result).reshape(-1, model.n)


def snapshot_condition(model, price_beliefs, queues):
    """Condition each query independently; do not feed snapshots into history."""
    if len(price_beliefs) != len(queues): raise ValueError("mismatched queries")
    out = []
    for belief, (bid, ask) in zip(price_beliefs, queues):
        i = model.index(bid, ask, -1)
        out.append(normalize(belief[i:i + 2]))
    return np.array(out).reshape(-1, 2)
