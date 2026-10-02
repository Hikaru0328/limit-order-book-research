# Queue depletion and short-horizon price formation

Read theory.md, then notebooks/queue_race.ipynb. The other notebooks isolate one
mechanism at a time: pure cancellation, market consumption, replenishment,
censoring, cancellation-rate sensitivity, and jump dependence.

Each notebook is a compact English experiment with fresh synthetic outputs.
The reusable code is in src/queue_models and its tests in tests/queue_checks.
Run from the repository root using scripts/run_notebooks.py.
