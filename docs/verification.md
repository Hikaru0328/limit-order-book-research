# Release verification

Verified on 2026-10-02 using Python 3.12.0.

- Editable package installation succeeded in an isolated virtual environment using existing numerical dependencies.
- All **97 unit tests passed** through the installed package: 44 queue-model, 38 memory/prediction and 15 continuous-time tests.
- All seven English notebooks executed successfully from cleared outputs.
- Full linear-Gaussian baseline rerun: five training/test seed pairs, 65,536 training events and 32,768 test events each.
- Maximum linear population identity discrepancy: 1.78e-15.
- Full continuous-time baseline rerun: 144 independent paths, nine models, 1,024 scoring time units.
- Saved-result audit passed; selected seeded replay discrepancy: 2.78e-17.
- Small preliminary checks were replaced by the full baseline outputs; they are not pooled with them.
- Additional finite-LOB and price-prediction full experiment drivers were not rerun in this release. Their core modules have unit-test coverage.

## Environment

| Package | Version |
|---|---|
| numpy | 1.26.0 |
| scipy | 1.13.1 |
| pandas | 2.1.1 |
| matplotlib | 3.9.2 |
| nbformat | 5.9.2 |
| nbclient | 0.8.0 |
| ipykernel | 6.30.1 |

## Scope

The publication checker screens text for common credential, resource, personal-path and conversation-identifier patterns; it also checks notebook execution and parses Python files. It is not a guarantee of secret detection or mathematical correctness. All nine regenerated figure panels/files were visually reviewed; no personal or operational content was identified.

Tests validate specified synthetic models and numerical implementations. No empirical market-validation claim follows from passing these checks.
