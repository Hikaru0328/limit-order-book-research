# Memory under partial observation

The linear-Gaussian benchmark separates exact projected memory from finite-sample
estimation. Endpoint observations of the complete state provide a Markov control;
partial observations and block means need not remain Markov.

The finite LOB adds queue imbalance, removal imbalance and price changes. Exact
finite-state moments and complete-basis controls distinguish information loss
from a restricted linear representation. Fixed-horizon price-prediction drivers
use validation-only ridge selection and independent Markov reference runs.

Start with the small linear benchmark in docs/reproduction.md. The finite-LOB and
price-prediction full drivers are supplied for deeper reproduction, but their
large historical outputs are not included. No closed RG dynamics or market alpha
is claimed. Full protocols can be computationally expensive.
