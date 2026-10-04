# Horizon experiment: model, measurements, and evidence

This note supports the [portfolio README](../README.md). The dense horizon
experiment is implemented in the private research repository in
[Draft PR #1](https://github.com/Hikaru0328/lob-memory-coarse-graining/pull/1),
commit `f897322d51b8208952abe5230d8154b3e7e69ce8`.
This public note includes figures and numerical summaries. The full dense-sweep
implementation and path-level outputs remain in the private research repository;
links to that repository require access.

## 1. Model and parameters

The continuous-time Markov state is X=(q_bid,q_ask,z), where queues have cap 8 and
z is a hidden binary direction in {-1,+1}. Price is an additive process: ask
queue depletion produces a +1 tick move, bid depletion a -1 tick move. Either
depletion resets both queues to 4. Spread is fixed.

With baseline arrival rate lambda=2, market-consumption rate mu=1, and per-order
cancellation rate theta=0.25, the event rates are:

| Event | Rate |
|---|---|
| Bid / ask limit arrival | lambda exp(+beta z) / lambda exp(-beta z); suppressed at cap |
| Bid / ask market consumption | mu exp(-beta z) / mu exp(+beta z) |
| Bid / ask cancellation | theta q_bid / theta q_ask |
| Hidden-direction flip | kappa |

Beta controls how strongly hidden direction changes arrival and consumption
intensities; it is not a time constant. Kappa is the symmetric flip rate.
The mean time to a flip is 1/kappa, while the autocorrelation time of z is
1/(2 kappa), because E[z_(t+s)|z_t]=z_t exp(-2 kappa s).

## 2. Mapping the README figure

| Case | beta | kappa | Hidden correlation time | Current-book maximum H | Marked-history mean maximum H | Oracle maximum H |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.5 | 0.02 | 25 | 5 | 7 | 8 |
| B | 0.5 | 0.2 | 2.5 | 1 | 2 | 2 |
| C | 0.5 | 2 | 0.25 | 1 | 1 | 1 |
| D | 1 | 0.02 | 25 | 4 | 4 | 5 |
| E | 1 | 0.2 | 2.5 | 1 | 2 | 2 |
| F | 1 | 2 | 0.25 | 1 | 1 | 1 |

All maxima refer to the integer grid 1–16. In Case A the current-book population
R² at H=4 and 5 is nearly identical. In Case D marked-history H=4–5 is near-flat.
In Case E the history mean prefers H=2, but H=2 minus H=1 is not resolved by the
paired pointwise interval. The figure omits three beta=0 null-control models:
they have no history advantage and peak at H=1 within the grid.

## 3. Target, information sets, and R²

The target is the cumulative price change Y_H=P_(t+H)-P_t, not a future queue
coordinate. For information F_t, the optimal known-parameter predictor is
m_F=E[Y_H|F_t]. Population predictability is

R²_F(H) = Var(m_F) / Var(Y_H)
        = 1 - E[(Y_H-m_F)^2] / Var(Y_H).

The denominator changes with H. A decrease in R² can occur while predictable
variance grows, if unpredictable variance grows faster. This is variance-normalized
predictability, not dollar profits or a risk-adjusted trading objective.

- Current book conditions on the current two queue sizes under the stationary law.
- Price history + current queues conditions on price history and the current
  snapshot; snapshots are not recursively fed into that price filter.
- Queue / price history includes event times, queue destinations, and price changes.
- Marked history additionally distinguishes market orders from cancellations.
- Full-state oracle knows current queues and z, but not future flips or events.

The overview figure shows current book, marked history, and oracle to keep the
main argument readable. The [full figure](../results/synthetic/horizon_sweep/all_observation_arms.png)
retains all four observation arms and the oracle.

Oracle and current-book curves are stationary population calculations. History
curves use the baseline conditional-expectation identity estimator: for information
including current queues, R²_F = R²_current + E[(m_F-m_current)^2]/Var(Y_H).
The latter expectation is estimated across paths. Direct-risk and realized-return
estimates are also exported; finite-sample versions need not coincide exactly.

The recoverable fraction, used elsewhere in the study, is
(R²_F-R²_current)/(R²_oracle-R²_current). It is not the fraction of all return
variance explained and is undefined when the denominator is zero.

## 4. Controlled comparison and uncertainty

Only the target horizon changes: H=1,2,...,16. The nine original beta/kappa settings,
16 independent paths per model, seeds, scoring origins, observation inputs, and
filtering methods are unchanged. Scoring duration is 1024 model time units, with
queries every 2 units after burn-in max(50,10/(2 kappa)). Simulation begins at the
stationary distribution; finite observed history is retained from time zero.
The maximum target horizon remains 16, so the paths and their endpoints match
baseline. Horizons 1,4,16 are replays, not fresh independent replications.

Intervals are Student-t 95% intervals across the 16 independent paths. Comparisons
between horizons pair the same paths. Overlapping within-path targets are not
counted as independent observations. Intervals are pointwise, not simultaneous
confidence sets for a selected peak.

Selected marked-history contrasts:

| Case | Contrast | Mean R² difference | Pointwise 95% interval |
|---|---|---:|---|
| A | H=8 minus H=7 | -0.000837 | [-0.001133, -0.000541] |
| B | H=2 minus H=1 | 0.005023 | [0.004461, 0.005586] |
| D | H=5 minus H=4 | -0.000066 | [-0.000471, 0.000340] |
| E | H=2 minus H=1 | 0.000473 | [-0.001081, 0.002027] |

A maximum at H=1 leaves shorter horizons untested. An interior grid maximum does
not establish the exact continuous optimum. No monotonic parameter law, causal
timescale decomposition, or closed RG flow has been established.

## 5. Evidence and reproduction

The private experiment directory is `studies/continuous_time_lob/experiments/07_horizon_sweep`.
Its README contains the exact commands and dependencies. Run, analyze, then audit;
the outputs include 144 population rows and 11,520 per-path prediction rows.

Recorded checks: 47 continuous-time tests passed; shared-horizon population
quantities and all seed/event records match baseline exactly; per-path predictions
match within 1.16e-14. Two paths independently replayed across all 16 horizons
match within 1.12e-16. Summary intervals and numerical clock/filter/null controls
also pass. Environment and source/result hashes are saved with the experiment.

Numerical evidence accompanying this release:

- [Discrete-grid maxima](../results/synthetic/horizon_sweep/grid_peaks.csv)
- [Adjacent paired horizon contrasts](../results/synthetic/horizon_sweep/adjacent_horizon_contrasts.csv)
- [Oracle variance decomposition](../results/synthetic/horizon_sweep/oracle_decomposition.csv)
- [Full observation-arm figure](../results/synthetic/horizon_sweep/all_observation_arms.png)

## 6. Relation to the other studies

Queue depletion establishes price-formation mechanisms. Gaussian controls validate
projection and estimation; the finite-LOB study distinguishes missing state
information from a restricted linear function span. These are different models,
with different clocks and reset rules. Their Mori kernels are not fitted prediction
coefficients, and a nonzero kernel alone does not demonstrate a forecasting gain.

The separate private [Gaussian horizon control](https://github.com/Hikaru0328/lob-memory-coarse-graining/pull/2)
finds no interior peak for its unchanged benchmark's cumulative-flow or endpoint
targets. This does not prove that an interior peak requires nonlinear dynamics.
