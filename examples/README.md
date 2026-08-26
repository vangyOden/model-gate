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
pip install "bdp-model-gate[structured]" jupyter

jupyter lab examples/bdp_model_gate_walkthrough.ipynb
```

Python 3.9–3.13 are all supported. The `structured` extra pulls
`scikit-learn`, `fairlearn` and `shap`; without it the checks that need them
report `NOT_APPLICABLE` instead of failing, which the notebook demonstrates.

> **Installing 0.2.0 specifically?** That release allowed
> `shap>=0.44,<0.47`, which cannot import against `numpy>=2.0` — a
> combination its `numpy<3.0` range permitted — so a fresh install raised
> `TypeError: Converting 'np.inexact' or 'np.floating' to a dtype not
> allowed`. 0.2.1 raised the floor to `shap>=0.48` and fixed it. Prefer
> 0.2.1 or later.

The committed outputs were produced on Python 3.13 against
`bdp-model-gate==0.2.1` installed from PyPI.
