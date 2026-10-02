# Validation specification: linear Mori memory under temporal coarse-graining

This specification governs the implemented benchmarks. It refines the exploratory
notes: neither a closed RG beta function nor market universality is assumed.

## 1. Scope, stationarity, and observations

The benchmark is a zero-mean stable linear-Gaussian process:

```math
x_{n+1}=Ax_n+\epsilon_{n+1},\qquad
\epsilon_n\overset{\mathrm{iid}}{\sim}N(0,Q),\qquad
\Sigma=A\Sigma A^\top+Q.
```

Innovations are independent of the initial state. Simulation starts from
$`N(0,\Sigma)`$. Both full-state observations and the first coordinate alone are used.
The leading observed coordinates must have nonsingular covariance.

Array block j contains samples jb through (j+1)b-1. Endpoint observations use
the last sample; sums and means use every sample. Only complete blocks are kept.
For theory, an equivalent stationary block can be indexed 1 through b.

For equal-length complete blocks, sum, mean and endpoint aggregation compose
exactly. For unequal lengths, means require counts. Ratios are never recursively
averaged: accumulate numerator components and denominators before dividing.
The LOB implementation retains both removal totals, endpoint queues, returns and counts.

The resolved output need not be sufficient to merge blocks and need not be Markov.
Associative raw aggregation does not prove closure of a reduced dynamical model.

## 2. Stochastic trajectories and projection

On the probability space of stationary full trajectories, let U be the shift:
$`Uf(\omega)=f(T\omega)`$. Block shifts are $`U_b=U^b`$. Aggregated observables
are functions of finite trajectory windows. The same ambient shift therefore
serves all scales, without pretending that a scale-specific reset accumulator
has an unchanged one-event transition rule.

Use the linear orthogonal Mori projection onto the centered components of Y:

```math
Pf=E[fY_0^\top]C_0^{-1}Y_0,\qquad C_k=E[Y_kY_0^\top].
```

For Gaussian linear observations this also coincides with conditional expectation
on linear observables. It is not the full conditional-expectation projection for
general nonlinear functions or the nonlinear LOB features.

Define coefficient matrices by the action on Y, not by identifying an abstract
operator with a matrix:

```math
\Omega_kY=P U(Q U)^kY,\qquad Q=I-P.
```

The finite-origin trajectory identity is

```math
Y_{n+1}=\sum_{k=0}^n\Omega_kY_{n-k}+W_n,\qquad E[W_nY_0^\top]=0.
```

W is not required to be white, independent, or orthogonal to every past Y.
A truncated stationary predictor is not automatically an optimal predictor.
Population origin orthogonality is tested with independent trajectory origins;
finite-sample fitted moment identities alone would be an insufficient check.

If instead using the state transition operator $`Kf=E[f(X_{n+1})\mid X_n]`$,
its Dyson identity is an identity of conditional expectations. A pathwise
formula additionally needs innovation terms or the trajectory-space construction.

## 3. Three independent routes to the same kernel

### Analytic elimination with the chosen projection

Partition x=(y,z). Put $`L=\Sigma_{zy}\Sigma_{yy}^{-1}`$ and r=z-Ly.
Transform A into the covariance-orthogonal coordinates:

```math
TAT^{-1}=\begin{pmatrix}F&B\\C&D\end{pmatrix},\qquad
T=\begin{pmatrix}I&0\\-L&I\end{pmatrix}.
```

Then

```math
\Omega_0=F,\qquad \Omega_k=BD^{k-1}C\quad(k\ge1).
```

For a full-state endpoint observation there are no hidden coordinates and all
memory coefficients vanish. For partial observations the covariance
orthogonalization is essential; setting the hidden state to zero defines a
different projection.

### Population covariance recursion

Compute correlations from the model, independently of the elimination:

```math
\Omega_0=C_1C_0^{-1},\qquad
\Omega_n=\left(C_{n+1}-\sum_{k=0}^{n-1}\Omega_kC_{n-k}\right)C_0^{-1}.
```

Endpoint block dynamics are

```math
A_b=A^b,\qquad Q_b=\sum_{r=0}^{b-1}A^rQ(A^r)^\top.
```

For block averages $`Y_j=b^{-1}\sum_{i=1}^b Cx_{jb+i}`$,

```math
C_k^{(b)}=b^{-2}\sum_{i=1}^b\sum_{j=1}^b C\Gamma(kb+i-j)C^\top,
\quad \Gamma(h)=A^h\Sigma\ (h\ge0),\quad\Gamma(-h)=\Gamma(h)^\top.
```

An independent exact construction lifts the state to [block average, endpoint].
Its transition is linear, and its joint noise covariance retains cross terms
because the average and endpoint share event innovations. Analytic elimination
in this lifted model must match the covariance-sum construction. This is tested
for block means and sums. At b=1 the lifted state can be redundant, so the
original endpoint model is used instead.

### Finite-sample estimation

Estimate the training mean and correlations, then apply the same recursion.
No ridge is used in the benchmark. Constant/redundant variables must be removed;
ill-conditioned C0 is rejected. The population target and the data estimator
therefore use the same projection convention.

## 4. Prediction is a separate experiment

OLS/VAR regressions jointly optimize coefficients on current and lagged states;
their coefficients are not labeled Mori kernels. Current-only and history fits
use identical training rows and identical independent test targets.

Two target conventions are exported: one block ahead, and 16 microscopic events
between observation endpoints. For block means, the averaging window still
changes with b; a fixed endpoint distance is not an identical physical target.
The LOB experiment reports pooled state-vector R2, not price-only alpha.

The memory cutoff is a 32-event gap from the latest predictor to the oldest one:
L=32/b on the chosen scales. Omega[k] connects Y_(n-k) to Y_(n+1), so the
predictor-to-target distance is (k+1)b, while the gap relative to Y_n is kb.
Reported memory strength is a finite-window sum, not an infinite-memory integral.
Near-zero strength makes the characteristic lag uninformative.

## 5. Units and scale flow

Under Y'=SY, matrices transform as $`\Omega'_k=S\Omega_kS^{-1}`$.
Raw Frobenius norms depend on coordinates. An additional training-covariance
whitening uses $`\widehat\Omega_k=C_0^{-1/2}\Omega_kC_0^{1/2}`$.
For invertible coordinate changes, the whitened representations differ by an
orthogonal similarity, so their Frobenius norms agree. Across scales this is a
specified convention, not proof of universality. Raw and whitened strengths
are both exported for the LOB.

Discrete coefficient matrices are not a continuous-time convolution density.
No b-squared rescaling is applied to them. The exported log-scale differences
are finite-step descriptive slopes, not an identified autonomous beta function.

A constructive nonclosure test uses

```math
A_\pm=\begin{pmatrix}.3&.25\\.2&\pm.4\end{pmatrix},\qquad
Q_\pm=I-A_\pm A_\pm^\top.
```

Both are stable, have positive innovation covariance, and stationary covariance I.
At b=1 their scalar Markov term, absolute memory strength and characteristic lag
are equal: .3, 1/12, and 5/3 (infinite-lag values). Their kernels differ in sign
pattern. At b=2 their memory strengths differ by a factor of 49. Therefore those
summaries, even with Omega0 and C0, do not close the scale flow. A richer closed
state remains a research question.

## 6. Boundaries of the claims

The linear controls validate algebra, estimator conventions and finite-sample
recovery. The minimal fixed-spread LOB checks pipeline connectivity only. It is
not calibrated; cancellation/market impact and quote resets are deliberately
simplified. Its linear Mori memory can reflect omitted nonlinear functions as
well as hidden states and temporal aggregation. These experiments establish
neither long-memory asymptotics, fixed points, universality, nor profitable alpha.

## Sources

- [Lin et al., regression-based Mori–Zwanzig projection](https://arxiv.org/abs/2205.05135)
- [Lin and Lu, data-driven reduction and Wiener projections](https://math.jhu.edu/~feilu/pub/LinLu20.pdf)

The explicit Gaussian benchmark, lift, aggregation records and nonclosure
counterexample above define this repository's validation construction.
