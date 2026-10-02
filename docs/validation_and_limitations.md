# Validation and limitations

Tests compare independent mathematical constructions, limiting cases, causal
feature alignment, exact population moments, and seeded Monte Carlo samples.
Numerical agreement validates these implementations within their assumptions.
It does not validate the assumptions against a real exchange.

- Single-queue examples separate restricted means from unrestricted means and
  retain censored paths. Infinite theoretical mean is not estimated by a finite mean.
- Replenishment correction uses the known model to estimate residual time; it is
  not data-only identification of an unobserved tail.
- Queue races assume independent consumption and independent resets. Direction and
  waiting time can be dependent within a cycle.
- Mori projection coefficients and fitted OLS/ridge prediction coefficients are
  different objects. Nonzero reduced memory need not imply non-Markov full dynamics.
- Exact finite-state calculations have floating-point and truncation limitations.
- Prediction selection uses training/validation splits. Overlapping target rows
  are not independent samples. Independent paths are the uncertainty units.
- Continuous-time filters know the generating parameters. Their equations may be
  exact while reported recoverability still has Monte Carlo uncertainty.
- No realistic variable-spread dynamics, calibration, profitability, live trading,
  universal scaling law, closed RG flow or optimal feature dimension is established.

The regenerated continuous-time baseline uses the full 16-path-per-model protocol. No claims from larger unshipped studies are reproduced
as validated findings here. See results/README.md for exactly what was rerun.
