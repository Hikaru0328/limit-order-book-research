# Reproduction

Install from the repository root with `python -m pip install -e '.[research]'` in
an isolated environment. Tests run with `python -m unittest discover -s tests -v`.
Set numerical threads to one for consistent runtimes:

```sh
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg
python scripts/run_notebooks.py
python studies/memory_and_prediction/experiments/linear_gaussian/run.py --output results/synthetic/linear_gaussian
python studies/continuous_time_lob/experiments/01_baseline/run.py --output results/synthetic/continuous_time
python studies/continuous_time_lob/experiments/01_baseline/audit.py --output results/synthetic/continuous_time
python studies/continuous_time_lob/experiments/01_baseline/plot.py --output results/synthetic/continuous_time
python scripts/check_publication.py
```

These commands regenerate the shipped outputs. The notebook runner executes with
its own Python interpreter and only replaces a notebook after successful execution.
Other experiment scripts write results incrementally; use a separate `--output`
directory for exploratory runs. Seeds are fixed, but floating-point details and
figures can differ across library versions and platforms.

A smaller check uses `--seeds 2 --samples 4096` for the linear benchmark and
`--replicas 2 --duration 16` for the continuous-time baseline. Write those results
under `outputs/`, not over the release examples.

Additional complete drivers, without inherited numerical outputs:

```sh
python studies/memory_and_prediction/experiments/toy_model/run.py --help
python studies/memory_and_prediction/experiments/finite_lob/run.py --help
python studies/memory_and_prediction/experiments/price_prediction/run.py --help
```

The price-prediction default includes 99 independent Markov reference runs and can
be substantially more expensive. Their presence is not a claim that all those
full protocols were rerun for this release. No compiled C extension is needed by
the selected baseline; later unshipped model extensions are outside this package.
