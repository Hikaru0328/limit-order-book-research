"""Marked finite-state CTMC with an additive reference-price process.

Rows are source states. Boundary LO arrivals are suppressed, not silently
observed as null events. Either kind of depletion resets BOTH queues.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.linalg import expm
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import expm_multiply

EVENTS = ("LO_bid", "LO_ask", "MO_bid", "MO_ask", "C_bid", "C_ask", "flip")


@dataclass(frozen=True)
class Parameters:
    cap: int = 8
    refill: int = 4
    lam: float = 2.0
    mu: float = 1.0
    theta: float = 0.25
    kappa: float = 0.2
    beta: float = 0.5
    tick: float = 1.0
    spread: float = 1.0

    def __post_init__(self):
        if not isinstance(self.cap, (int, np.integer)) or self.cap < 2:
            raise ValueError("cap must be integer >= 2")
        if not isinstance(self.refill, (int, np.integer)) or not 2 <= self.refill <= self.cap:
            raise ValueError("refill must be integer in [2, cap]")
        rates = np.array([self.lam, self.mu, self.theta, self.kappa, self.tick, self.spread])
        if not np.all(np.isfinite(rates)) or np.any(rates <= 0):
            raise ValueError("rates, tick and spread must be finite and positive")
        if not np.isfinite(self.beta) or not 0 <= self.beta <= 10:
            raise ValueError("beta must lie in [0, 10]")

    def slowed(self, scale):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("positive finite scale required")
        return replace(self, **{k: getattr(self, k) / scale for k in ("lam", "mu", "theta", "kappa")})


class Model:
    def __init__(self, parameters=Parameters()):
        self.parameters = p = parameters
        self.states = np.array([(b, a, z) for b in range(1, p.cap + 1)
                                for a in range(1, p.cap + 1) for z in (-1, 1)])
        self.n = len(self.states)
        self.rates = np.empty((self.n, 7))
        self.targets = np.empty((self.n, 7), int)
        self.rewards = np.zeros((self.n, 7))
        for i, (b, a, z) in enumerate(self.states):
            up, down = np.exp(p.beta * z), np.exp(-p.beta * z)
            self.rates[i] = [p.lam * up * (b < p.cap), p.lam * down * (a < p.cap),
                             p.mu * down, p.mu * up, p.theta * b, p.theta * a, p.kappa]
            for e in range(7):
                bb, aa, zz = b, a, z
                if e == 0: bb = min(b + 1, p.cap)
                elif e == 1: aa = min(a + 1, p.cap)
                elif e in (2, 4): bb -= 1
                elif e in (3, 5): aa -= 1
                else: zz = -z
                if bb == 0 or aa == 0:
                    self.rewards[i, e] = p.tick * (1 if aa == 0 else -1)
                    bb = aa = p.refill
                self.targets[i, e] = self.index(bb, aa, zz)
        self.events = np.zeros((7, self.n, self.n))
        for e in range(7):
            self.events[e, np.arange(self.n), self.targets[:, e]] = self.rates[:, e]
        self.generator = self.events.sum(0) - np.diag(self.rates.sum(1))
        self.price_jumps = {}
        for sign in (-1, 1):
            mat = np.zeros((self.n, self.n))
            for e in range(6):
                np.add.at(mat, (np.arange(self.n), self.targets[:, e]),
                          self.rates[:, e] * (self.rewards[:, e] == sign * p.tick))
            self.price_jumps[sign] = mat
        self.reward_operator = p.tick * (self.price_jumps[1] - self.price_jumps[-1])
        self.square_reward_operator = p.tick ** 2 * (self.price_jumps[1] + self.price_jumps[-1])
        eq = self.generator.T.copy(); eq[-1] = 1
        rhs = np.zeros(self.n); rhs[-1] = 1
        self.stationary = np.linalg.solve(eq, rhs)
        if self.stationary.min() <= 0 or np.max(abs(self.stationary @ self.generator)) > 1e-10:
            raise ArithmeticError("stationary distribution failed")

    def index(self, bid, ask, z):
        return int(2 * ((bid - 1) * self.parameters.cap + ask - 1) + (z == 1))

    def transition(self, time):
        if not np.isfinite(time) or time < 0: raise ValueError("nonnegative time required")
        return expm(time * self.generator)

    def price_moments(self, horizon):
        """Conditional first and second moments of future cumulative price.

        g'=L g + R1, h'=L h + 2R g + R2 1, g(0)=h(0)=0.
        """
        if not np.isfinite(horizon) or horizon < 0: raise ValueError("nonnegative horizon required")
        l = csr_matrix(self.generator); r = csr_matrix(self.reward_operator)
        a = csr_matrix((self.reward_operator @ np.ones(self.n))[:, None])
        b = csr_matrix((self.square_reward_operator @ np.ones(self.n))[:, None])
        zero = csr_matrix((self.n, self.n)); bottom = csr_matrix((1, self.n))
        aug = bmat([[l, zero, a], [2 * r, l, b], [bottom, bottom, csr_matrix((1, 1))]], format="csr")
        initial = np.zeros(2 * self.n + 1); initial[-1] = 1
        result = expm_multiply(horizon * aug, initial)
        return result[:self.n], result[self.n:2 * self.n]

    def current_queue_belief(self, bid, ask):
        i = self.index(bid, ask, -1)
        v = self.stationary[i:i + 2]
        return v / v.sum()

    def quotes(self, mid):
        mid = np.asarray(mid)
        return mid - self.parameters.spread / 2, mid + self.parameters.spread / 2


@dataclass
class Path:
    initial: int
    times: np.ndarray
    event: np.ndarray
    state: np.ndarray
    reward: np.ndarray
    end: float

    def states_at(self, times):
        times = np.asarray(times)
        if np.any(times < 0) or np.any(times > self.end): raise ValueError("time outside path")
        return np.r_[self.initial, self.state][np.searchsorted(self.times, times, side="right")]

    def prices_at(self, times):
        times = np.asarray(times)
        if np.any(times < 0) or np.any(times > self.end): raise ValueError("time outside path")
        return np.r_[0., np.cumsum(self.reward)][np.searchsorted(self.times, times, side="right")]


def simulate(model, end, seed=0):
    """Direct Gillespie sampler; hidden switches remain separate events."""
    if not np.isfinite(end) or end <= 0: raise ValueError("positive end required")
    rng = np.random.default_rng(seed)
    state = initial = int(rng.choice(model.n, p=model.stationary))
    t = 0.; times = []; events = []; states = []; rewards = []
    while True:
        rates = model.rates[state]; total = rates.sum()
        t += rng.exponential(1 / total)
        if t > end: break
        e = int(np.searchsorted(np.cumsum(rates), rng.random() * total, side="right"))
        rewards.append(model.rewards[state, e]); state = model.targets[state, e]
        times.append(t); events.append(e); states.append(state)
    return Path(initial, np.array(times), np.array(events, int), np.array(states, int),
                np.array(rewards), float(end))
