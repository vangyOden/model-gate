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
python3 -m venv .venv && source .venv/bin/activate

pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "bdp-model-gate[structured]" jupyter
jupyter lab examples/bdp_model_gate_walkthrough.ipynb
```

### Two install notes

**1. TestPyPI needs `--extra-index-url`.** TestPyPI does not mirror `numpy`,
`pandas`, `scikit-learn`, `fairlearn` or `shap`, so the bare
`pip install -i https://test.pypi.org/simple/ bdp-model-gate` cannot resolve
dependencies. `--extra-index-url https://pypi.org/simple/` lets pip fall back
to real PyPI for them.

**2. `shap` pins (fixed in 0.2.1).** In 0.2.0 and earlier the `structured`
extra allowed `shap>=0.44,<0.47`, which cannot import against `numpy>=2.0` —
a combination the `numpy<3.0` range permitted. A fresh install produced shap
0.46 with numpy 2.x and raised:

```
TypeError: Converting `np.inexact` or `np.floating` to a dtype not allowed
```

0.2.1 raises the floor to `shap>=0.48`, which imports cleanly against numpy
2.x and ships cp313 wheels, so no manual pinning is needed and Python
3.9–3.13 all work. If you are installing **0.2.0**, pin a numpy-1 stack in
the *same* pip command as the package (installing first and downgrading
afterwards can leave numpy's compiled libraries broken):

```bash
pip install "numpy==1.26.4" "scipy==1.13.1" "scikit-learn==1.5.2" \
            "pandas==2.2.3" "shap==0.46.0"
```

The committed outputs were produced on Python 3.13 against
`bdp-model-gate==0.2.1` (built from this repository — 0.2.1 is not on
TestPyPI yet) with shap 0.49.1 and numpy 2.5.2. An earlier pass was
validated against `0.2.0` installed from TestPyPI to confirm the published
package works.
