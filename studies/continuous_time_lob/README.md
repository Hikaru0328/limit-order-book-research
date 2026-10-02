# Continuous-time hidden-flow inference

A finite-state generator combines unit limit arrivals, market orders, per-order
cancellation and a hidden binary direction. Hidden direction modulates order
intensities. Either queue depleting moves price and resets both queues.

The baseline compares current queues, price history, queue/price history and
marked event history. Filters know the parameters. Marks, event timing and
survival between events can carry different information. Hidden switch timestamps
are not supplied to observed-history filters.

Spread is fixed, queue size is capped, and all parameters are synthetic. Recovering
information relative to a current-queue baseline is not recovering every hidden
state or forecasting real prices. The full-state oracle is a model-specific ceiling.

Only the one-direction baseline is included in this release. Slow/fast hidden
modes, learned feature compression and larger sensitivity studies are outside its
validated scope. See docs/reproduction.md for the small release run and full run.
