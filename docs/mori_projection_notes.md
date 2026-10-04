# Mori projection, effective memory, and the finite-LOB experiment

This note explains the discrete, linear Mori projection used in this repository.
The central question is how eliminating observables creates history dependence,
even when the microscopic dynamics are Markov. The finite-LOB experiment also
illustrates a subtler possibility: all microscopic coordinates can be known,
while the chosen **linear span of functions** still generates Mori memory.

The derivation below fixes the operator ordering, distinguishes conditional
expectations from stochastic trajectories, and connects the mathematics to the
[finite-LOB projection bridge](../studies/memory_and_prediction/experiments/finite_lob_bridge/README.md).
Implementation and committed results were checked against revision
[`0e21514`](https://github.com/Hikaru0328/limit-order-book-research/commit/0e2151449676e74cf1069af160de2b9eef8bd8d0)
on 2026-10-04. This is an explanation of an existing synthetic experiment, not
an extension of its scientific implementation.

## 1. Why memory appears after coarse-graining

Suppose the microscopic state is Markov:

```math
\Pr(X_{t+1}\mid X_t,X_{t-1},\ldots)=\Pr(X_{t+1}\mid X_t).
```

Observe only a vector

```math
Y_t=g(X_t).
```

Different microscopic states can produce the same value of $Y_t$. Their next-step
distributions may differ, and past observations can help distinguish which state
is currently occupied. Consequently, the law of $Y_{t+1}$ given $Y_t$ need not
coincide with its law given the entire observed past. A many-to-one observation
can produce non-Markov dynamics, although special projections can remain Markov.

The physical picture is a delayed return of information:

```math
\text{resolved observables}
\longrightarrow \text{unresolved observables}
\longrightarrow \text{resolved observables}.
```

Instead of tracking all intermediate variables, a reduced equation accounts for
their later influence through memory and a residual. This is the dynamical
analogue of integrating out degrees of freedom in statistical mechanics or field
theory: the retained description can acquire interactions nonlocal in time.
The exact rewriting does not itself supply a computational closure; unresolved
dynamics still have to be evaluated or approximated.

A second distinction will matter below. Even if $Y_t$ determines $X_t$, a small
**linear** predictor space may not contain the nonlinear state functions needed
for its evolution. Linear Mori memory then concerns that function space, rather
than necessarily demonstrating non-Markov behavior of the observed coordinates.

## 2. Notation and the two clocks

All observable vectors are columns. Expectations use a stationary probability
law, and observables entering covariance and projection formulas are centered.

| Symbol | Meaning |
|---|---|
| $X_t$ | Full microscopic Markov state |
| $Y_t$ | Resolved observable vector, or its centered version when taking moments |
| $\mathcal U$ | One-step evolution operator acting on state functions |
| $\mathcal P$ | Linear Mori projection onto the resolved function span |
| $\mathcal Q=1-\mathcal P$ | Complementary, unresolved projection |
| $C_k$ | Future–past covariance $E[Y_{t+k}Y_t^\top]$ |
| $\Omega_k$ | Mori coefficient matrix in the chosen feature coordinates |
| $T$ | Readout from those coordinates to the common target $(I,D,R)$ |
| $K_k$ | Covariance-normalized common-target coefficient |
| $N_t^b,N_t^a$ | Bid and ask queue quantities |

Code and earlier documentation use `qb`, `qa`, or $Q_b,Q_a$ for queues. Here we
use $N^b,N^a$ so that quantities cannot be confused with $\mathcal Q$. Removal
**counts** within a block will instead be written $A_j,B_j$.

In the abstract derivation, one step means one update of the process under
consideration. For the queue chain it means one microscopic event. For block
observations we use a block index $j$, a block size $b$, and one step means
$b$ events. We do not identify these two time units.

## 3. The evolution operator acts on functions

For a state function $f$, define

```math
(\mathcal U f)(x)=E[f(X_{t+1})\mid X_t=x].
```

This is a linear operator even when the state dynamics or observable $f$ are
nonlinear. Iteration and the Markov property give

```math
(\mathcal U^n f)(x)=E[f(X_{t+n})\mid X_t=x].
```

If a finite transition matrix $P_{\rm tr}$ has source states in rows, a column of
function values evolves as $P_{\rm tr}f$. A row probability distribution evolves
as $\mu P_{\rm tr}$. The same matrix occurs, but these are different actions:
$\mathcal U$ in this note evolves **observables**, not probability densities.

For deterministic dynamics $X_{t+1}=\Phi(X_t)$, the definition reduces to the
Koopman action $\mathcal U f=f\circ\Phi$. For a stochastic chain,
$\mathcal U^nY(X_t)$ is a conditional mean, not generally the realized $Y_{t+n}$.
Section 5 keeps this distinction explicit.

Stationarity makes the mean-zero subspace invariant under $\mathcal U$. No
reversibility, detailed balance, or equilibrium fluctuation–dissipation identity
is assumed in the covariance recursion used here.

## 4. Mori projection is a least-squares projection of functions

Let $Y=(Y_1,\ldots,Y_d)^\top$ be centered and square-integrable, with nonsingular

```math
C_0=E[YY^\top].
```

For a scalar or vector-valued state function $f$, project each component by

```math
\mathcal P f=E[fY^\top]C_0^{-1}Y,\qquad
\mathcal Q f=f-\mathcal P f.
```

To see why, minimize $E[\|f-BY\|^2]$ over constant matrices $B$. The normal equations
are $BC_0=E[fY^\top]$, so the minimizer is precisely the coefficient above.
With the stationary $L^2$ inner product this is the orthogonal projection onto

```math
\mathcal S=\mathrm{span}\{Y_1,\ldots,Y_d\}.
```

It obeys $\mathcal P^2=\mathcal P$, $\mathcal Q^2=\mathcal Q$,
$\mathcal P\mathcal Q=0$, and

```math
E[(\mathcal Q f)Y^\top]=0.
```

Thus $\mathcal P f$ is the part representable by the resolved features, while
$\mathcal Q f$ contains everything outside their linear span. The features can
themselves be nonlinear functions of $X$; the projection remains linear in their
coefficients. Constants have already been removed by centering. For uncentered
functions, one would also retain an intercept or a constant basis function.

“Unresolved” can mean missing microscopic coordinates, or missing functions of
coordinates that are already known. For example, retaining a quantity $N$ does
not put every function of $N$ in $\mathrm{span}\{N\}$. A conditional-expectation
projection $E[f\mid Y]$ instead allows all square-integrable functions of $Y$;
it is generally different from the finite linear Mori projection used here.
The Polynomial versus Complete comparison tests exactly this distinction.

## 5. Deriving the discrete Mori equation

### 5.1 First steps and ordering

Operators compose from right to left. They act componentwise on vector functions;
constant coefficient matrices commute with the scalar evolution action. Start with

```math
\mathcal UY=\mathcal P\mathcal UY+\mathcal Q\mathcal UY
           =\Omega_0Y+F_1,
\qquad F_1=\mathcal Q\mathcal UY.
```

Evolve again, and split the evolved unresolved function:

```math
\begin{aligned}
\mathcal U^2Y
 &=\Omega_0\mathcal UY+\mathcal UF_1\\
 &=\Omega_0\mathcal UY+\Omega_1Y+F_2,\\
\Omega_1Y&=\mathcal P\mathcal UF_1,
\qquad F_2=\mathcal Q\mathcal UF_1.
\end{aligned}
```

One further step gives

```math
\begin{aligned}
\mathcal U^3Y
 &=\Omega_0\mathcal U^2Y+\Omega_1\mathcal UY+\mathcal UF_2\\
 &=\Omega_0\mathcal U^2Y+\Omega_1\mathcal UY+\Omega_2Y+F_3,
\end{aligned}
```

where $\Omega_2Y=\mathcal P\mathcal UF_2$ and $F_3=\mathcal Q\mathcal UF_2$.
Inductively, with $F_0=Y$,

```math
F_{k+1}=\mathcal Q\mathcal UF_k=(\mathcal Q\mathcal U)^{k+1}Y,
\qquad
\Omega_kY=\mathcal P\mathcal U(\mathcal Q\mathcal U)^kY,
```

and hence

```math
\boxed{\mathcal U^{n+1}Y
=\sum_{k=0}^{n}\Omega_k\mathcal U^{n-k}Y+F_{n+1}.}
```

The matrix itself is

```math
\Omega_k=
E\!\left[(\mathcal U(\mathcal Q\mathcal U)^kY)Y^\top\right]C_0^{-1}.
```

The operator $\mathcal P\mathcal U(\mathcal Q\mathcal U)^k$ is not literally a
$d\times d$ matrix on the entire function space. Its action on $Y$ defines the
matrix $\Omega_k$ in the displayed convention. Swapping $\mathcal Q\mathcal U$
for $\mathcal U\mathcal Q$, or transposing coefficients without also changing the
covariance convention, would generally change the expression.

For $k\ge1$ its path is

```math
Y\xrightarrow{\mathcal U}\cdot\xrightarrow{\mathcal Q}\cdot
\xrightarrow{\mathcal U}\cdot\xrightarrow{\mathcal Q}\cdots
\xrightarrow{\mathcal U}\cdot\xrightarrow{\mathcal P}\Omega_kY.
```

There are $k$ unresolved projections followed by a return to the resolved span.
This is the sense in which $\Omega_k$ describes effective feedback returning from
the unresolved sector after $k$ memory lags. $F_{n+1}$ is the part that has not
returned. “Orthogonal dynamics” means repeated $\mathcal Q\mathcal U$ evolution;
it is not a separate Markov process on hidden queue coordinates.

In [finite_lob.py](../src/lob_memory/finite_lob.py), `direct_event_mori` implements
this order directly. It evolves the current function array, projects onto the
centered features, records the coefficient transpose required by row-stored
samples, and retains the unresolved difference for the next step.

### 5.2 What the equation means on stochastic trajectories

For deterministic evolution the preceding identity immediately becomes a
trajectory equation. For a stochastic Markov chain it first describes conditional
means. To write the corresponding exact finite-origin trajectory relation, define

```math
F_{n+1}^{(t)}
:=Y_{t+n+1}-\sum_{k=0}^{n}\Omega_kY_{t+n-k}.
```

Then

```math
\boxed{Y_{t+n+1}
=\sum_{k=0}^{n}\Omega_kY_{t+n-k}+F_{n+1}^{(t)}.}
```

Often the superscript is suppressed and the final term is simply called
$F_{n+1}$. Here it is essential to distinguish that **full trajectory residual**
from the state function $F_{n+1}=(\mathcal Q\mathcal U)^{n+1}Y$:

```math
E[F_{n+1}^{(t)}\mid X_t]=F_{n+1}(X_t),\qquad
E[F_{n+1}^{(t)}Y_t^\top]=0.
```

The difference contains the randomness of future transitions. It does not vanish
merely because the feature span is closed under conditional-mean evolution.
This is consistent with `origin_residuals` in [mori.py](../src/lob_memory/mori.py):
its output indexed by $n$ is called $W_n$ in the code and corresponds to
$F_{n+1}^{(t)}$ here. It is orthogonal to the **original** resolved vector $Y_t$,
not necessarily to every observation between $t$ and $t+n$. It need not be white
noise or a one-step innovation, and its finite-origin index matters.

## 6. The covariance recursion in the repository

Define the future–past covariance in exactly the implemented orientation:

```math
C_k=E[Y_{t+k}Y_t^\top]
   =E[(\mathcal U^kY)(X_t)Y_t^\top].
```

Multiply either version of the Mori equation on the right by $Y_t^\top$ and take
expectations. Since $\mathcal PF_{n+1}=0$, the residual term vanishes:

```math
C_{n+1}=\sum_{k=0}^{n}\Omega_kC_{n-k}.
```

The term containing the new coefficient is $\Omega_nC_0$. Moving all earlier
terms to the left gives

```math
\boxed{\Omega_n=
\left(C_{n+1}-\sum_{k=0}^{n-1}\Omega_kC_{n-k}\right)C_0^{-1}.}
```

In particular,

```math
\begin{aligned}
\Omega_0&=C_1C_0^{-1},\\
\Omega_1&=(C_2-\Omega_0C_1)C_0^{-1},\\
\Omega_2&=(C_3-\Omega_0C_2-\Omega_1C_1)C_0^{-1}.
\end{aligned}
```

The subtraction removes propagation already accounted for by the instantaneous
term and the shorter-lag memory terms. It is an operator decomposition inferred
from two-time moments, not successive fits of a growing autoregression.
For vectors, matrix order matters: $\Omega_kC_{n-k}$ cannot generally be replaced
by $C_{n-k}\Omega_k$.

`from_covariances` in [mori.py](../src/lob_memory/mori.py) implements this recursion
without ridge regularization. Its transposed linear solve computes right
multiplication by $C_0^{-1}$. `covariance_sequence` uses centered samples to estimate
$C_k$ as `z[k:].T @ z[:-k] / (N-k)` for positive $k$. Such sample covariances and
the resulting kernels have estimation error. In contrast, the headline bridge
uses exact finite-state population moments, subject only to numerical arithmetic
and the explicitly controlled lag truncation.

## 7. AR/VAR versus Mori: different projection spaces

A population VAR of order $p$ chooses all matrices jointly to minimize one-step
squared prediction error:

```math
Y_{t+1}=A_0Y_t+A_1Y_{t-1}+\cdots+A_{p-1}Y_{t-p+1}+\epsilon_{t+1}.
```

Its predictor space contains all components of the lag-augmented vector,

```math
\mathrm{span}\{Y_t,Y_{t-1},\ldots,Y_{t-p+1}\}.
```

Without regularization, the optimal population residual is orthogonal to every
included lagged predictor. Whiteness additionally requires suitable model
assumptions; it does not follow from finite-order least squares alone.

Linear Mori first fixes the same-time function space $\mathcal S$, projects the
rest into $\mathcal Q$, and follows how evolved unresolved functions return.
The central difference is the projection space and the question being asked,
not simply simultaneous regression versus sequential fitting.

| Aspect | AR/VAR | Linear Mori in this repository |
|---|---|---|
| Objective | Best linear prediction using specified lags | Decompose evolution relative to a specified resolved function span |
| State / predictor space | Components of a lag-augmented vector | Same-time functions $\mathcal S$; their complement is propagated |
| Coefficients | Joint least-squares $A_j$ for a chosen order | $\Omega_k$ from projection and the covariance recursion |
| Interpretation | Partial predictive contributions conditional on the other included lags | Return of unresolved function dynamics to the resolved span |
| Typical use | Forecasting and residual diagnostics | Reduced dynamics, representation comparisons, memory diagnostics |
| Prediction focus | Directly optimizes a finite-history prediction problem | An exact identity including a residual; truncation or forecasting needs additional choices |
| Coarse-graining interpretation | Not intrinsic to the fit | Explicitly depends on what functions and scales are retained |

AR/VAR asks whether history improves prediction. Mori asks how unresolved degrees
of freedom generate history dependence in a reduced description. Neither question
subsumes all of the other. In particular, setting the Mori residual to zero and
truncating its kernel does not automatically produce the optimal VAR predictor.

There are connections between generalized Langevin and ARMA representations,
but kernel conventions and residual restrictions matter. The time-series analysis
of [Niemann et al. (2008)](https://doi.org/10.1103/PhysRevE.77.011117) is a useful
warning against treating an arbitrary memory-kernel shape as a direct diagnostic
of familiar time-series properties.

## 8. An AR(2) example makes the distinction explicit

Consider a stable causal AR(2), with independent, zero-mean innovations of finite
positive variance:

```math
Y_{t+1}=a_1Y_t+a_2Y_{t-1}+\varepsilon_{t+1}.
```

The two-dimensional state $(Y_t,Y_{t-1})$ is Markov. Resolve only the scalar $Y_t$.
Multiplying by $Y_t$ and using stationarity and innovation orthogonality yields

```math
C_1=a_1C_0+a_2C_1,\qquad
\Omega_0=\frac{C_1}{C_0}=\frac{a_1}{1-a_2}.
```

Thus $\Omega_0$ generally differs from the AR coefficient $a_1$. The previous
observation is unresolved, but its correlation with the current one contributes
to the instantaneous least-squares projection. It has not simply been set to zero.

For additional intuition, write $r=C_1/C_0$. The next covariance obeys
$C_2=a_1C_1+a_2C_0$, so the Mori recursion gives

```math
\Omega_1=a_2(1-r^2).
```

For example, $a_1=0.5,a_2=0.2$ gives $r=0.625$ and $\Omega_1=0.121875$.
The subsequent scalar Mori coefficients need not terminate after the second lag,
even though the original autoregression has order two.

Now retain the augmented state:

```math
Z_t=\begin{pmatrix}Y_t\\Y_{t-1}\end{pmatrix},\qquad
Z_{t+1}=A Z_t+\begin{pmatrix}\varepsilon_{t+1}\\0\end{pmatrix},
\qquad A=\begin{pmatrix}a_1&a_2\\1&0\end{pmatrix}.
```

Here $\mathcal UZ=AZ$ lies exactly in the linear span of the two retained
coordinates. Therefore $\mathcal Q\mathcal UZ=0$, $\Omega_0=A$, and all higher
Mori coefficients vanish. Innovations remain in realized trajectories.

This is conditional-mean closure, not a claim that two coordinates span every
square-integrable function of a continuous state. **Markovness alone is not enough
to guarantee zero linear Mori memory.** The AR(2) augmentation has linear closure;
the Polynomial LOB representation below generally does not. The finite-state
indicator representation does.

## 9. The finite-LOB process and its three representations

### 9.1 Microscopic queue dynamics

The discrete finite LOB has state

```math
X_t=(N_t^b,N_t^a),\qquad N_t^b,N_t^a\in\{1,\ldots,8\}.
```

There are 64 states. Bid and ask additions each have probability 0.2; bid and ask
removals each have probability 0.3. Quantities change by one. A capped addition
is a self-transition and still counts as an event. When a removal depletes a
queue, **only that side** is refilled to 8, and the opposite quantity is retained.
Ask depletion moves price by $+1$ tick, bid depletion by $-1$ tick. Spread is fixed.
Market consumption and cancellation have identical queue effects and are combined.

These are the rules in `event_model`, not the different simultaneous-reset rules
of the continuous-time study; see [model assumptions](model_assumptions.md).
Absolute price is not needed to predict the next queue transition or signed price
increment, and need not be part of this finite stationary state.

### 9.2 Block observables are not all endpoint-state functions

For nonoverlapping block $j$ of $b$ events, let $N_j^b,N_j^a$ be its endpoint
quantities, $A_j,B_j$ its ask- and bid-removal counts, and $r_t$ its individual
signed price increments. Define

```math
I_j=\frac{N_j^b-N_j^a}{N_j^b+N_j^a},\qquad
D_j=\begin{cases}
(A_j-B_j)/(A_j+B_j),&A_j+B_j>0,\\
0,&A_j+B_j=0,
\end{cases}
\qquad
R_j=\sum_{t\text{ in block }j}r_t.
```

Thus a single ask removal gives $D=+1$, a bid removal gives $D=-1$, and an addition
gives $D=0$ at $b=1$. A removal need not move price. $D$ is a removal imbalance,
not an indicator that the queue actually reached zero. At larger $b$, it is the
ratio of aggregated counts, not the average of one-event removal ratios.

Only $I_j$ is a function of the endpoint queue alone. To apply the state-function
derivation literally to $D_j,R_j$, lift the process to a state $\widehat X_j$
containing the completed block path (or its start state and event list). This
lifted process is Markov: the next block depends on the present block only through
its endpoint. For $b=1$, the lift is the 256-state pair (pre-event queue, event
label), precisely the construction in `direct_event_mori`. The microscopic queue
chain still has 64 states; these are not 256 different queue configurations.

One may then write $Y_j=g(\widehat X_j)$ and use the same projection algebra on the
lifted stationary law. The block dynamic program computes its needed moments
without explicitly enumerating every lifted state. This avoids incorrectly
writing $(I,D,R)$ as a function of the 64-state endpoint alone.

### 9.3 Baseline, Polynomial, and Complete

Before centering, the three feature vectors are

```math
Y^{\rm baseline}=(I,D,R)^\top,
```

```math
Y^{\rm polynomial}=
\left(I,D,R,\frac{N^b}{8},\frac{N^a}{8},
\left(\frac{N^b}{8}\right)^2,
\left(\frac{N^a}{8}\right)^2,\frac{N^bN^a}{64}\right)^\top,
```

and 63 independent endpoint-state indicators followed by $D,R$ for Complete.
Their centered dimensions are respectively 3, 8, and 65.

Baseline loses the absolute endpoint quantities and most state functions.
Polynomial includes both quantities, so it identifies the endpoint microscopic
state exactly. Nevertheless, its eight-dimensional linear span need not contain
all functions produced by conditional evolution. Remaining Mori memory is not
proof that another microscopic coordinate is missing.

Complete spans every **centered endpoint-state function**, and additionally retains
$D,R$. It does not span every function of an entire block path. That is unnecessary
for this null control: the conditional expectation of any next-block feature,
given the present lifted state, is a function of the present endpoint. Complete
can represent each such function, giving

```math
\mathcal U\mathcal S_{\rm complete}\subseteq\mathcal S_{\rm complete},
\qquad \mathcal Q\mathcal UY^{\rm complete}=0.
```

Therefore its population $\Omega_k$ vanish for $k\ge1$, although stochastic
next-block fluctuations remain. This is the precise closure property tested.

## 10. Reconstructing imbalance from indicators

For the uncentered endpoint indicators,

```math
e_{n_b,n_a}=\mathbf 1\{N^b=n_b,N^a=n_a\},\qquad
I=\sum_{n_b=1}^{8}\sum_{n_a=1}^{8}
\frac{n_b-n_a}{n_b+n_a}e_{n_b,n_a}.
```

Imbalance is nonlinear in the quantities, but linear in the indicators. On a
finite set, a function is reconstructed by weighting each indicator with its
value at that state.

All 64 indicators sum to one. After centering, their sum is zero, so one must be
omitted to obtain an invertible covariance. The implementation orders states by
bid quantity then ask quantity and omits the last state, $(8,8)$.
Its imbalance coefficient is zero. Consequently,

```math
I-E[I]=\sum_{(n_b,n_a)\ne(8,8)}
\frac{n_b-n_a}{n_b+n_a}
\left(e_{n_b,n_a}-E[e_{n_b,n_a}]\right).
```

More generally, a centered function $f$ has coefficients $f(n_b,n_a)-f(8,8)$ in
the 63-indicator basis. `feature_map` uses this subtraction for queue polynomials;
for imbalance it reduces to the expression above. “Omitting one indicator” does
not discard information once centering or a constant term is handled correctly.

## 11. From representation-specific Omega to common-target K

$\Omega_k$ is the coefficient in a representation's own feature space. It is
$3\times3$ for Baseline, $8\times8$ for Polynomial, and $65\times65$ for Complete.
Raw matrix norms would mix different coordinate scalings, dimensions, and output
variables. The bridge instead compares their action on the same centered target
$Y_*=(I,D,R)^\top=TY$.

For Baseline and Polynomial,

```math
T_{\rm baseline}=I_3,\qquad
T_{\rm polynomial}=\begin{pmatrix}
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0
\end{pmatrix}.
```

Let $c\in\mathbb R^{63}$ list the imbalance coefficients for the retained states.
Then

```math
T_{\rm complete}=\begin{pmatrix}
c^\top&0&0\\
0_{1\times63}&1&0\\
0_{1\times63}&0&1
\end{pmatrix}.
```

With $C_*=TC_0T^\top$, the normalized comparison coefficient is

```math
\boxed{K_k=C_*^{-1/2}T\Omega_kC_0^{1/2}.}
```

The roots are the symmetric positive-definite covariance roots, as used in
`target_kernel` in [finite_lob.py](../src/lob_memory/finite_lob.py).
To understand their orientation, introduce whitened coordinates

```math
\widehat Y=C_0^{-1/2}Y,\qquad
\widehat Y_*=C_*^{-1/2}Y_*.
```

A lag contribution in physical coordinates is $T\Omega_kY$.
Since $Y=C_0^{1/2}\widehat Y$, its expression in whitened output coordinates is
$K_k\widehat Y$. The factor on the **right is the positive square root**, because
it converts a whitened input back to physical feature coordinates. It is not
$C_0^{-1/2}$. The left factor whitens the common output.

| Representation | $T$ | $C_0$ and $\Omega_k$ | $C_*$ | $K_k$ |
|---|---|---|---|---|
| Baseline | $3\times3$ | $3\times3$ | $3\times3$ | $3\times3$ |
| Polynomial | $3\times8$ | $8\times8$ | $3\times3$ | $3\times8$ |
| Complete | $3\times65$ | $65\times65$ | $3\times3$ | $3\times65$ |

The $K_k$ are therefore generally **rectangular**. For a unit-covariance input,
$E[\|K_k\widehat Y\|^2]=\|K_k\|_F^2$, which explains the covariance weighting.
It does not make lag contributions statistically independent. The common target
and normalization permit a meaningful comparison of these couplings, but do not
imply that adding features must reduce their summed norms.

An invertible linear change of coordinates within the same resolved span changes
its whitened coordinates by an orthogonal transformation, leaving these
Frobenius norms unchanged. Enlarging the span is a different operation.
$K_k$ introduces no new physical mechanism: it expresses the same Mori feedback
in normalized input and common-output coordinates.

## 12. Memory diagnostics and their time units

Only $k\ge1$ is included in memory; $\Omega_0$ is the instantaneous contribution.
Define

```math
n_k=\|K_k\|_F,\qquad
M=\sum_{k\ge1}n_k,
```

and, when the denominator is nonzero and the sums converge,

```math
\theta=\frac{\sum_{k\ge1}k n_k}{\sum_{k\ge1}n_k},\qquad
\xi_b=b\theta_b.
```

Here $n_k$ measures the normalized effective coupling at lag $k$; $M$ sums that
strength over memory lags. $\theta$ is a strength-weighted lag in **block units**,
and $\xi_b$ expresses that lag gap in **microscopic event units**.
In the equation predicting block $j+1$, the term $\Omega_kY_{j-k}$ is $k$ blocks
behind the current block but $k+1$ blocks behind the predicted block. Accordingly,
the CSV distinguishes `event_horizon = k*b` from `target_distance = (k+1)*b`.

[rg.py](../src/lob_memory/rg.py) computes the finite-array versions after the bridge
has removed index zero with `common[1:]`. Its docstring describes square kernels,
but the implementation only requires a three-dimensional array and also accepts
the rectangular common-target matrices above. This is a documentation shorthand,
not a numerical restriction; no scientific code change is needed.

These diagnostics are not incremental $R^2$, significance tests, or proofs of
long memory. They ignore coefficient signs when taking norms, and history vectors
at different lags are correlated. Summing squared norms is not a decomposition
of forecast variance, either.

A true null has no characteristic lag: the mathematical ratio is $0/0$.
The utility returns zero for an exactly zero array as a computational convention.
For tiny roundoff residuals the bridge retains the raw ratio but sets
`horizon_resolved=False`; those apparent horizons have no physical interpretation.

## 13. What the validated finite-LOB results show

### 13.1 Population results at one event per block

The committed [summary CSV](../studies/memory_and_prediction/experiments/finite_lob_bridge/results/summary.csv)
contains the following values. These are deterministic population calculations,
not Monte Carlo estimates or fits to market observations.

| Representation | Dimension | $M$ | $\theta$ | $\xi$ in events |
|---|---:|---:|---:|---:|
| Baseline | 3 | 0.2201996537 | 5.52315459 | 5.52315459 |
| Polynomial | 8 | 0.0590646727 | 3.41173172 | 3.41173172 |
| Complete | 65 | $3.8665\times10^{-16}$ | Unresolved numerical null | Unresolved numerical null |

Baseline has appreciable projection-induced memory under this normalization.
The tested polynomial expansion reduces its total by about 73%, despite already
recovering the endpoint coordinates. Complete removes it to numerical precision.
Together these controls support **projection-induced memory in this synthetic
model**, including memory caused by a restricted function span.

The [lag plot](../studies/memory_and_prediction/experiments/finite_lob_bridge/results/projection_memory.svg)
displays $n_k$, not a cumulative sum. The non-null kernels decay with lag in this
experiment; memory is not “growing as more lags are added.” The distinct cumulative
quantity

```math
M(L)=\sum_{j=1}^{L}\|K_j\|_F
```

is nondecreasing by definition even when every late-lag contribution is small.

### 13.2 How population construction and tail control work

`block_population` sums over event paths using a dynamic program, retaining
removal counts and first and second return moments. It constructs $C_0$ and the
cross-block moments without sampling. As an independent ordering check,
`direct_event_mori` computes projection coefficients on the lifted event chain at
$b=1$; these agree with the covariance recursion within the recorded tolerance.

For longer lags, [projection_bridge.py](../src/lob_memory/projection_bridge.py)
uses centered factors $L,R\in\mathbb R^{64\times d}$ such that

```math
C_\ell=R^\top(P_b^\top)^{\ell-1}L,\quad\ell\ge1,\qquad P_b=P_{\rm tr}^{\,b}.
```

Here $L,R$ are auxiliary factor matrices, not the scalar block return or a Mori
projector. Centering gives $\mathbf1^\top L=0$ and $R^\top\pi=0$. Set

```math
A=P_b^\top-\pi\mathbf1^\top-LC_0^{-1}R^\top.
```

Substitution into the recursion gives

```math
\Omega_k=R^\top A^kLC_0^{-1},\qquad k\ge0.
```

The stationary-mode subtraction does not alter centered covariances. If
$K_k=\mathsf F A^k\mathsf G$, the code checks $q=\|A^{32}\|_2<1$ and uses

```math
\sum_{k>L_0}\|K_k\|_F
\le \|\mathsf F A^{L_0+1}\|_F
\frac{\sum_{j=0}^{31}\|A^j\mathsf G\|_2}{1-q}.
```

An analogous lag-weighted bound controls $\theta$. These are geometric truncation
bounds under exact arithmetic, evaluated numerically; they do not cover roundoff
and are not statistical confidence intervals. At $b=1$, the stored Baseline and
Polynomial sums use 128 memory lags, with remaining-strength bounds approximately
$1.35\times10^{-16}$ and $1.55\times10^{-16}$ respectively.

The Complete factor-based total is about $3.9\times10^{-16}$ at $b=1$. The
independent covariance-recursion calculation can have larger roundoff: its
largest 32-lag sum over all tested scales is about $1.26\times10^{-14}$.
Both are below the fixed numerical-null threshold $10^{-11}$. The complete
residuals were not forcibly zeroed. See the committed
[verification record](../studies/memory_and_prediction/experiments/finite_lob_bridge/results/verification.json).

### 13.3 Scale dependence and finite-sample estimation

| Block size $b$ | Baseline $M_b$ | Polynomial $M_b$ | Baseline $\xi_b$ | Polynomial $\xi_b$ |
|---:|---:|---:|---:|---:|
| 1 | 0.22019965 | 0.05906467 | 5.52315 | 3.41173 |
| 2 | 0.19107275 | 0.03545401 | 5.76358 | 4.24587 |
| 4 | 0.13377121 | 0.01874471 | 6.54106 | 5.56329 |
| 8 | 0.06604503 | 0.01020587 | 9.22725 | 8.35383 |
| 16 | 0.02606755 | 0.00073909 | 16.39484 | 16.09921 |

The reported strength decreases while the event-unit lag increases. At coarse
scales $\theta_b$ approaches one block, but one block contains more events. That
combination does not establish a growing physical tail or asymptotic long memory.
The ordering Baseline ≥ Polynomial ≥ Complete holds at these tested scales; it
is an observed result, not a general theorem about feature enrichment.

If covariances are instead estimated from a finite trajectory, their errors enter
every stage of the recursion. Inverting an ill-conditioned $C_0$ can amplify them.
A nonzero estimated Complete kernel must therefore be compared with its population
null and sampling uncertainty; it is not automatically microscopic memory.
The separate [finite-sample driver](../studies/memory_and_prediction/experiments/finite_lob/run.py)
addresses that question and is not the source of the headline population values.

## 14. Relation to coarse-graining and RG

The statistical-mechanical analogy is

```math
\text{integrate out degrees of freedom}
\longrightarrow \text{effective dynamics nonlocal in time}.
```

The bridge adds an explicit scale $b$ through endpoint sampling and block
aggregation of removals and returns. It studies $M_b,\theta_b,\xi_b$ as functions
of that scale: **scale-dependent effective dynamics**.

The endpoint transition is $P_{\rm tr}^{\,b}$, but the block observables $D,R$ also
change with $b$. Their moments require the block construction; replacing the
transition by a matrix power alone does not fully specify the coarse observation.
Similarly, discarding function directions at a fixed $b$ and changing $b$ are
distinct interventions.

The current work has not established a closed RG transformation, beta functions
for a closed set of couplings, a fixed point, universality, or empirical market
scaling. `logscale_beta` in `rg.py` can compute finite differences $\Delta g/\Delta
\log b$ for supplied diagnostic values. That operation alone does not demonstrate
an autonomous flow $dg/d\log b=\beta(g)$ independent of omitted information.
The projection identity is a starting point for investigating such questions,
not a completed renormalization-group theory.

## 15. Common misconceptions and implementation checks

**Memory is not autocorrelation.** An AR(1) has nonzero autocorrelations at many
lags, while its correctly specified scalar linear Mori memory vanishes beyond
$\Omega_0$. The recursion subtracts correlations explained by resolved propagation.

**A kernel norm is not incremental $R^2$.** It measures a normalized coupling.
Predictive benefit requires a specified target, comparison predictor, and risk
calculation, including how residuals and truncation are handled.

**Mori is not sequential AR fitting.** The same-time projection space and
orthogonal-dynamics construction determine the coefficients. A lag-augmented
regression uses different normal equations and residual orthogonality.

**Nonzero memory need not mean missing state coordinates.** It can mean missing
functions of known coordinates. Polynomial is the explicit example here.
Knowing $N^b,N^a$ is not the same as having every function of them in a finite
linear feature span.

**Complete does not eliminate stochastic noise.** Its endpoint indicators span
all centered queue-state functions needed for next-block conditional expectations.
This closes the chosen span and removes higher Mori coefficients. Random future
events still generate trajectory residuals.

**A coarse-unit horizon is not proof of physical long memory.** Convert block
lags to event units, check the tail and strength, and exclude numerical-null
ratios before interpreting a characteristic timescale.

**Verification scope.** The equations were checked against `mori.py`,
`finite_lob.py`, `projection_bridge.py`, and `rg.py`, together with the model
assumptions and committed bridge tables. Dimensions, ordering, centering,
imbalance readout, signs, block-count ratios, and the exclusion of $\Omega_0$
follow the code. All 45 tests in the existing memory test suite passed, and an isolated $b=1$
bridge reproduction matched the committed values. Independent checks using an
explicit 256-state transition and projection matrix verified the first three
operator identities (largest absolute discrepancy below $2.2\times10^{-14}$).
A stationary AR(2) covariance calculation reproduced $\Omega_0=0.625$ and
$\Omega_1=0.121875$, with higher augmented-state coefficients below
$1.1\times10^{-16}$. Readout dimensions and the covariance-weighted norm identity
were also checked. Published experiment outputs were not replaced.

The main bridge README and implementation agree on the scientific results.
Two distinctions are made more explicit here: block observables require a lifted
state for a state-function derivation, and a stochastic trajectory residual is
not just $(\mathcal Q\mathcal U)^nY(X_t)$. The square-shape wording in `rg.py`
does not describe the rectangular matrices actually used in the common-target
comparison. None of these clarifications requires a change to the calculation.

## 16. References

These references provide background, not a claim that the finite LOB reproduces
their physical models. Bibliographic metadata were checked against publisher
records, original papers, or author-submitted arXiv records on 2026-10-04.
The discrete signs and ordering in this note are defined in Sections 5–6;
continuous-time kernels in other conventions need not have identical signs.

1. R. Zwanzig, “Memory Effects in Irreversible Thermodynamics,” *Physical Review*
   **124**, 983–992 (1961).
   [Publisher record / DOI](https://doi.org/10.1103/PhysRev.124.983).
   A foundational projection treatment of delayed response and reduced dynamics.
2. H. Mori, “Transport, Collective Motion, and Brownian Motion,” *Progress of
   Theoretical Physics* **33**(3), 423–455 (1965).
   [DOI](https://doi.org/10.1143/PTP.33.423);
   [original paper](https://courses.physics.ucsd.edu/2020/Fall/physics210b/Mori-1965.pdf).
   The linear projection viewpoint underlying the terminology used here.
3. R. Zwanzig, *Nonequilibrium Statistical Mechanics*, Oxford University Press
   (2001). [Publisher record](https://academic.oup.com/book/52730).
   A textbook introduction, including projection operators and nonlinear problems.
4. H. Grabert, *Projection Operator Techniques in Nonequilibrium Statistical
   Mechanics*, Springer Tracts in Modern Physics **95**, Springer (1982).
   [Publisher record / DOI](https://doi.org/10.1007/BFb0044591).
   A systematic treatment of projection-operator methods beyond this fixed linear setup.
5. M. te Vrugt and R. Wittkowski, “Projection operators in statistical mechanics:
   a pedagogical approach,” *European Journal of Physics* **41**, 045101 (2020).
   [DOI](https://doi.org/10.1088/1361-6404/ab8e28);
   [author manuscript and journal metadata](https://arxiv.org/abs/2001.01572).
6. Y. T. Lin, Y. Tian, M. Anghel, and D. Livescu, “Data-driven learning for the
   Mori-Zwanzig formalism: a generalization of the Koopman learning framework,”
   arXiv:2101.05873v3 (2021).
   [Verified preprint version](https://arxiv.org/abs/2101.05873v3).
   Cited here specifically in its preprint version for the data-driven Mori/Koopman connection.
7. M. Niemann, T. Laubrich, E. Olbrich, and H. Kantz, “Usage of the Mori-Zwanzig
   method in time series analysis,” *Physical Review E* **77**, 011117 (2008).
   [Publisher record / DOI](https://doi.org/10.1103/PhysRevE.77.011117).
   Examines discrete memory kernels, their interpretation, and their relation to ARMA models.
