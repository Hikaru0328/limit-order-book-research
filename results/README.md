# Regenerated synthetic results

All outputs here were freshly generated for this public release. No historical
result directory was copied. Numerical units are model units, not calibrated market time.

| Experiment | Release settings | Evidence |
|---|---|---|
| Seven queue notebooks | Each notebook displays its seed and sample count; queue race uses 100,000 cycles per condition | Executed cells, tables and PNG figures |
| Linear-Gaussian benchmark | Five training/test seed pairs; 65,536 training and 32,768 test events per pair | Population kernels, convergence, independent prediction results and figure |
| Continuous-time baseline | Nine models, 16 independent paths each; 1,024 scoring time units | Parameters, seeds, path-level results, summaries, seeded replay audit and figure |

Software versions are recorded with the runs. The continuous-time audit compares
saved summaries, exact clock controls, selected path replays and numerical filter
controls. Hashes identify the audited files; they do not establish market validity.

Intervals and seed dispersion must be interpreted according to the experiment.
They are not simultaneous guarantees across all displayed comparisons. The linear
one-block target varies with block size; fixed-event-target data are separate.
See docs/verification.md for the checks performed on this extraction.
