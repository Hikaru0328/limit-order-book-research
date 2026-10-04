# From queue events to price moves

For a queue of size q with independent per-order cancellation rate theta and no
arrivals, depletion time is the maximum of q exponential lifetimes. Equivalently,
it is the sum of independent exponential holding times with rates k theta,
for k=1,...,q. Its mean is H_q/theta and variance is
sum(1/k^2)/theta^2. Adding unit market consumption changes each death rate to
mu + k theta.

With unit arrival rate lambda, generator action before absorption is

$$Lf(q)=\lambda[f(q+1)-f(q)]+(\mu+\theta q)[f(q-1)-f(q)].$$

Mean hitting times satisfy Lu=-1 with u(0)=0, but admissibility and behavior at
infinity matter. The implementation compares a positive series with a tail bound,
a finite boundary-value solve, and uniformization. With no cancellation and equal
positive birth/death rates, absorption occurs almost surely but its mean time is
infinite. A finite observation window instead estimates E[min(T,c)] and P(T>c).

For independent ask and bid consumption, starting at i and j units, set
p=mu_ask/(mu_ask+mu_bid). The next upward-move probability is

$$P(\text{ask first})=\sum_{k=0}^{j-1}{i+k-1\choose k}p^i(1-p)^k.$$

Independent resets produce iid cycle pairs (J,T), while J and T within a cycle
may be dependent. Drift is v=E[J]/E[T]. Under the light-tail assumptions used here,
the long-run variance rate is

$$\sigma^2=\frac{\mathrm{Var}(J)+v^2\mathrm{Var}(T)-2v\mathrm{Cov}(J,T)}{E[T]}.$$

Omitting the covariance term can either increase or decrease the inferred rate.
All statements refer to these synthetic assumptions; spread competition and
empirical directional forecasting are separate questions.
