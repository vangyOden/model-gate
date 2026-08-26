# Examples

Five notebooks, each committed **with outputs** so they read on GitHub
without being run. None needs external data or credentials.

| Notebook | Task | Domain | Model |
|---|---|---|---|
| [01 binary classification](01_binary_classification_sklearn.ipynb) | binary | credit scoring | `GradientBoostingClassifier` |
| [02 multiclass and ordinal](02_multiclass_ordinal_sklearn.ipynb) | multiclass, ordinal | underwriting: accept / refer / decline | `RandomForestClassifier` |
| [03 regression](03_regression_sklearn.ipynb) | regression | motor premium, claims severity, claims frequency | `GradientBoostingRegressor` |
| [04 PyTorch and friends](04_any_framework_classification.ipynb) | binary | fraud | PyTorch, Keras-shaped, remote endpoint |
| [05 boosters and the CLI](05_boosters_and_cli.ipynb) | binary | fraud | XGBoost `XGBClassifier` and `Booster` |

**Start with 01.** It covers the core machinery — contexts, checks, reports,
verdicts, configuration, custom checks, plugins, validation and the CLI. The
others assume it and focus on what their task or framework changes.

## Running them

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "bdp-model-gate[structured]" jupyter

jupyter lab examples/
```

Notebooks 01–03 need nothing beyond that. The other two need one extra
package each:

| Notebook | Extra | Note |
|---|---|---|
| 04 | `pip install torch` | CPU wheel is enough |
| 05 | `pip install xgboost` | on macOS also `brew install libomp`, which XGBoost requires on that platform |

Python 3.9–3.13 are all supported.

### Why PyTorch and XGBoost are in separate notebooks

Not a stylistic choice. On macOS the two link different OpenMP runtimes and
**segfault when used in the same process** — a hard crash, not an exception.
A single notebook covering both would die partway through for many readers,
so the frameworks are kept apart.

## Re-executing them

The notebooks are validated manually at release time rather than in CI. To
re-run them all and fail on the first error:

```bash
./examples/run_all.sh
```

That executes each notebook in place with `nbconvert` and reports any cell
that raised. Run it whenever the library's behaviour changes — a notebook
committed with stale outputs is worse than no notebook, because the outputs
look authoritative.

The committed outputs were produced on Python 3.13 with
`bdp-model-gate` 0.4.1, scikit-learn 1.7, shap 0.49, torch 2.13 and
xgboost 3.4.
