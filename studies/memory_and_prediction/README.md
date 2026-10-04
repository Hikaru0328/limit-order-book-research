# Memory under partial observation

The linear-Gaussian benchmark separates exact projected memory from finite-sample
estimation. Endpoint observations of the complete state provide a Markov control;
partial observations and block means need not remain Markov.

The [finite-LOB projection bridge](experiments/finite_lob_bridge/README.md) transfers
this idea to the existing Markov queue model before continuous-time hidden-flow
inference. It compares three projection spans using exact population memory,
complete-state null controls and a descriptive block-size sweep. Polynomial
features restore the queue coordinates but do not span every nonlinear function.

For a self-contained derivation and interpretation, see the
[Mori projection research note](../../docs/mori_projection_notes.md), including
the AR/VAR distinction and the finite-LOB common-target kernel.

The finite LOB adds queue imbalance, removal imbalance and price changes. Exact
finite-state moments and complete-basis controls distinguish information loss
from a restricted linear representation. Fixed-horizon price-prediction drivers
use validation-only ridge selection and independent Markov reference runs.

Start with the small linear benchmark in docs/reproduction.md. The older finite-sample recovery and
price-prediction full drivers are supplied for deeper reproduction, but their
large historical outputs are not included. No closed RG dynamics or market alpha
is claimed. Full protocols can be computationally expensive.
