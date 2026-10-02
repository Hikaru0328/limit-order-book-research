# Models and assumptions

These are separate models with different reset rules, not successive versions of
one calibrated simulator.

| Model | Events | Depletion/reset | Spread |
|---|---|---|---|
| Single queue | Unit arrivals, removals, per-order cancellation, depending on experiment | Zero absorbing | No price or spread |
| Queue race | Independent unit Poisson consumption on bid and ask | Fixed quantities independently reset after each price jump | Not modeled |
| Discrete finite LOB | Unit additions and removals with fixed probabilities | Only depleted side refills; other side retained | Fixed one tick |
| Continuous-time LOB | Separate arrival, market, cancellation and hidden-direction rates | Both queues reset on either depletion | Fixed |
| Linear-Gaussian control | Stable linear dynamics with Gaussian innovations | Not a market simulator | Not applicable |

The discrete model counts capped addition attempts as self-transitions. The
continuous-time model suppresses additions at the cap. Their clocks and boundary
behavior are therefore not interchangeable. Cancellation and market removal have
the same queue effect in the discrete model; marked continuous-time histories
can distinguish them.

Queue imbalance is (bid quantity - ask quantity)/(bid quantity + ask quantity).
Do not identify this observable with the full state or with a calibrated price
probability. Time aggregation can create effective memory even from Markov dynamics.
