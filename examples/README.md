# Examples

## `bdp_model_gate_walkthrough.ipynb`

An end-to-end tour of the library against a synthetic Nigerian credit-scoring
model, covering every public surface: the four check categories, configurable
performance metrics, `GateConfig` tuning, custom checks, the plugin entry
point, input validation, and the CLI with its three-way exit code.

The notebook is committed **with outputs**, so it reads on GitHub without
being run. It needs no external data or credentials.

### Running it

```bash
python3.12 -m venv .venv && source .venv/bin/activate

# One resolution pass — see gotcha 2 for why the pins go in the same command.
pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "bdp-model-gate[structured]==0.2.0" \
  "numpy==1.26.4" "scipy==1.13.1" "scikit-learn==1.5.2" \
  "pandas==2.2.3" "shap==0.46.0" \
  jupyter
jupyter lab examples/bdp_model_gate_walkthrough.ipynb
```

### Two install gotchas

**1. TestPyPI needs `--extra-index-url`.** TestPyPI does not mirror `numpy`,
`pandas`, `scikit-learn`, `fairlearn` or `shap`, so the bare
`pip install -i https://test.pypi.org/simple/ bdp-model-gate` cannot resolve
dependencies. `--extra-index-url https://pypi.org/simple/` lets pip fall back
to real PyPI for them.

**2. The `structured` extra currently resolves to a broken combination.**
`shap>=0.44,<0.47` is not compatible with `numpy>=2.0`, but `pyproject.toml`
allows `numpy<3.0`. A fresh install therefore picks up numpy 2.x and shap
0.46, and `import shap` raises:

```
TypeError: Converting `np.inexact` or `np.floating` to a dtype not allowed
```

Until the pins are tightened, install a numpy-1 stack **in the same pip
command** as the package (installing first and downgrading afterwards can
leave numpy's compiled libraries broken):

```bash
pip install "numpy==1.26.4" "scipy==1.13.1" "scikit-learn==1.5.2" \
            "pandas==2.2.3" "shap==0.46.0"
```

Also note `shap<0.47` publishes no wheels for Python 3.13 — use Python ≤ 3.12
if you want `ShapSubgroupCheck` to run. Without shap the check reports
`NOT_APPLICABLE` and the rest of the gate is unaffected.

The notebook's outputs were produced on Python 3.12 with that pinned stack,
against `bdp-model-gate==0.2.0` installed from TestPyPI.
