# Examples

Five notebooks, each executed and committed with outputs. None needs external
data or credentials.

| Notebook | Task | Domain | Model |
|---|---|---|---|
| [01 Binary classification](01_binary_classification_sklearn.ipynb) | binary | credit scoring | `GradientBoostingClassifier` |
| [02 Multiclass and ordinal](02_multiclass_ordinal_sklearn.ipynb) | multiclass, ordinal | underwriting | `RandomForestClassifier` |
| [03 Regression](03_regression_sklearn.ipynb) | regression | premium, severity, frequency | `GradientBoostingRegressor` |
| [04 PyTorch and friends](04_any_framework_classification.ipynb) | binary | fraud | PyTorch, Keras-shaped, remote |
| [05 Boosters and the CLI](05_boosters_and_cli.ipynb) | binary | fraud | XGBoost, `--model-loader` |

**Start with 01.** It covers the core machinery — contexts, checks, reports,
verdicts, configuration, custom checks, plugins, validation and the CLI. The
others assume it and focus on what their task or framework changes.

## Running them

```bash
pip install "bdp-model-gate[structured]" jupyter
jupyter lab examples/
```

01–03 need nothing more. 04 needs `torch`; 05 needs `xgboost` (and
`brew install libomp` on macOS, which XGBoost requires there regardless).

!!! note "Why PyTorch and XGBoost are in separate notebooks"
    On macOS the two link different OpenMP runtimes and **segfault in the same
    process** — a hard crash, not an exception. A single notebook covering
    both would die partway through for many readers.

Source lives in
[`examples/`](https://github.com/vanjy-eng/model-gate/tree/main/examples), and
`examples/run_all.sh` re-executes them all and fails on the first error.
