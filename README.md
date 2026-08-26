# BDP Model Gate

[![PyPI](https://img.shields.io/pypi/v/bdp-model-gate)](https://pypi.org/project/bdp-model-gate/)
[![Python versions](https://img.shields.io/pypi/pyversions/bdp-model-gate)](https://pypi.org/project/bdp-model-gate/)
[![License: MIT](https://img.shields.io/pypi/l/bdp-model-gate)](LICENSE)

Automated pre-deployment ML model governance: fairness, performance,
compliance, and security checks, run as a single gate that gives you a
`PASS` / `NEEDS_REVIEW` / `BLOCKED` status to wire into CI before a model
is promoted to production.

Currently covers **structured data models**. Unstructured (text, image,
audio) support is planned — see `bdp_model_gate.unstructured` for the reserved
interface and roadmap notes.

## Install

Available on [PyPI](https://pypi.org/project/bdp-model-gate/):

```bash
# core (context/report/gate objects only — no check logic that needs ML libs)
pip install bdp-model-gate

# structured-data checks (fairlearn, shap, scikit-learn) — install this for real use
pip install bdp-model-gate[structured]

# for running the test suite
pip install bdp-model-gate[dev]
```

Compliance and security checks (model card validation, adversarial
robustness, PII scanning, prompt-injection testing) work with just the core
install. Fairness checks need `fairlearn`/`shap`, and every performance
metric except `accuracy` needs `scikit-learn` — install the `structured`
extra to get all of it. On a core-only install the default `metric="auto"`
falls back to `accuracy` and says so loudly; see
[Choosing the performance metric](#choosing-the-performance-metric).

## Quickstart

```python
from bdp_model_gate import StructuredGateContext, ModelGate

context = StructuredGateContext(
    model=my_model,
    X=X_val,
    y_true=y_val,
    y_pred=y_pred,
    protected_df=protected_val,  # optional — enables fairness checks
    latencies_ms=benchmark_latencies,  # optional — enables performance checks
    cost_per_inference=0.0008,  # optional
    model_card=my_model_card,  # optional — enables compliance checks
    generate_fn=None,  # optional — set if there's a generative side-car
)

report = ModelGate().run(context)
print(report.summary())
report.to_json("gate_report.json")

if report.gate_status == "BLOCKED":
    raise SystemExit("Model failed governance gate — see gate_report.json")
```

Or the one-liner:

```python
from bdp_model_gate import run_structured_gate

report = run_structured_gate(model, X_val, y_val, y_pred, protected_df=protected_val)
```

## What each category checks

**Fairness** (non-blocking by default — routes to `NEEDS_REVIEW`, since some
flags need human judgment)
- `ProxyCorrelationCheck` — input features that correlate with a protected attribute
- `DisparateImpactCheck` — outcome-level demographic parity
- `ShapSubgroupCheck` — features whose SHAP contribution differs across groups
- `CounterfactualFlipCheck` — prediction shift when a protected attribute is flipped

**Performance** (blocking)
- `PerformanceThresholdCheck` — model score on a metric you choose, p95
  latency, cost-per-inference. See [Choosing the performance metric](#choosing-the-performance-metric).

**Compliance** (blocking)
- `ComplianceMappingCheck` — model card completeness, DPIA trigger for
  high-risk use cases, explainability requirement for models affecting a person

**Security** (blocking)
- `AdversarialRobustnessCheck` — prediction flip rate under small feature perturbation
- `PIILeakageCheck` — regex scan of string columns for PII patterns
- `PromptInjectionCheck` — canned jailbreak prompts against any generative side-car

## Customizing thresholds

```python
from bdp_model_gate import GateConfig
from bdp_model_gate.structured import default_structured_checks
from bdp_model_gate import ModelGate

config = GateConfig()
config.performance.metric = "roc_auc"
config.performance.min_score = 0.85
config.fairness.disparity_threshold = 0.05

gate = ModelGate(checks=default_structured_checks(config))
report = gate.run(context)
```

## Choosing the performance metric

`PerformanceConfig.metric` decides what the model is scored on, and
`min_score` is the threshold that score must clear. Set the two together —
`min_score` means nothing on its own.

```python
config = GateConfig()
config.performance.metric = "f1"  # what to measure
config.performance.min_score = 0.75  # what it has to beat
```

Built-in names: `roc_auc`, `average_precision`, `accuracy`,
`balanced_accuracy`, `f1`, `precision`, `recall`. All except `accuracy`
require scikit-learn (the `structured` extra).

**Label-based metrics need hard classes.** `accuracy`, `balanced_accuracy`,
`f1`, `precision`, and `recall` binarize continuous `y_pred` at
`config.performance.decision_threshold` (default `0.5`). Predictions already
in `{0, 1}` are left alone. Ranking metrics (`roc_auc`,
`average_precision`) use the raw scores and ignore the threshold.

**Your own metric.** Any `fn(y_true, y_pred) -> float` works, and is called
with `y_pred` exactly as you supplied it — no thresholding, since only you
know what your metric expects:

```python
from sklearn.metrics import fbeta_score


def f2(y_true, y_pred):
    return fbeta_score(y_true, (y_pred >= 0.3).astype(int), beta=2)


config.performance.metric = f2  # reported under the name "f2"
```

**`"auto"` (the default)** uses `roc_auc` when scikit-learn is installed and
falls back to `accuracy` when it isn't. The fallback is never silent: it's
logged at `WARNING`, marked `metric_is_fallback: true` in the result
metadata, and spelled out in the check's detail string. A score is only
comparable to `min_score` if you know which metric produced it, so the
report always names it:

```json
{
  "gate_status": "PASS",
  "model_metric": "roc_auc",
  "model_score": 0.9132
}
```

Naming a metric explicitly opts out of fallback entirely — if
`metric="roc_auc"` can't run, the gate reports a blocking `CHECK_ERROR`
rather than quietly scoring you on something else. A typo'd metric name
raises `GateConfigurationError` as soon as the check is constructed.

From the CLI, `--metric`, `--min-score`, and `--decision-threshold` do the
same thing, and take precedence over a `--config` file:

```bash
bdp-model-gate --model model.joblib --data validation.csv --target-col label \
  --metric f1 --min-score 0.75 --output gate_report.json
```

> **Migrating from 0.1.0:** `min_accuracy` is now `min_score`, and the old
> name was misleading — it was compared against ROC AUC whenever
> scikit-learn was installed, and accuracy otherwise. `min_accuracy` still
> works (in Python and in `--config` files) but emits a `DeprecationWarning`.
> Likewise `GateReport.model_auc` is superseded by `model_metric` /
> `model_score`, and now returns `None` unless the metric really was AUC.

## Writing your own check

```python
from bdp_model_gate import BaseCheck, CheckResult


class MyCustomCheck(BaseCheck):
    name = "my_custom_check"
    category = "compliance"  # fairness | performance | compliance | security
    blocking = True

    def run(self, context):
        # inspect context.model, context.X, context.model_card, etc.
        return [CheckResult(self.name, self.category, "OK", "looks fine", self.blocking)]


gate = ModelGate(checks=[MyCustomCheck()])
```

## Using it as a pre-deployment CI/CD gate

Installing the package gives you an `bdp-model-gate` console script, meant to
run as a **pre-deployment step** — after a model is trained/built, before
it's promoted to a registry or prod endpoint. It is not intended to run on
every PR.

```bash
bdp-model-gate \
  --model model.joblib \
  --data validation.csv \
  --target-col label \
  --protected protected.csv \
  --model-card model_card.json \
  --cost-per-inference 0.0008 \
  --output gate_report.json
```

Exit codes are chosen so a pipeline can distinguish three outcomes:

| Exit code | Status | Pipeline behavior |
|---|---|---|
| `0` | `PASS` | proceed to deploy automatically |
| `2` | `NEEDS_REVIEW` | stop and require a human sign-off (fairness flags need judgment) |
| `1` | `BLOCKED` | hard fail — performance, compliance, or security check failed |

A ready-to-adapt **Azure Pipelines** example is in
[`ci_examples/azure-pipelines.model-gate.yml`](ci_examples/azure-pipelines.model-gate.yml),
and a **GitHub Actions** equivalent (a reusable `workflow_call` workflow) is in
[`ci_examples/github-actions.model-gate.yml`](ci_examples/github-actions.model-gate.yml).
Both structure this as three stages/jobs: run the gate, a manual-approval
step gated behind exit code `2` (GitHub Environments / Azure Environments
with required reviewers), and a deploy step that only runs if the gate
passed outright or was manually approved. Point them at wherever your
training pipeline publishes `model.joblib` / `validation.csv` /
`protected.csv` / `model_card.json` as a build artifact.

Config overrides for the CLI can be JSON, YAML, or TOML — pick whichever
matches your repo's conventions:

```yaml
# config.yaml
performance:
  metric: f1
  min_score: 0.85
  decision_threshold: 0.5
fairness:
  disparity_threshold: 0.05
```

```bash
bdp-model-gate --model model.joblib --data validation.csv --target-col label \
  --config config.yaml --output gate_report.json
```

YAML configs need `pip install pyyaml` (or `bdp-model-gate[dev]`, which
already includes it); TOML needs `tomli` on Python < 3.11 (3.11+ has
`tomllib` built in).

Pass `-v`/`--verbose` for debug-level logging (per-check timing, which
checks ran/skipped and why) — the library uses the standard `logging`
module throughout, so it composes with whatever logging setup your
pipeline already has.

## Extending with plugins

Third-party packages can register additional checks without forking this
library, via the `bdp_model_gate.checks` entry-point group:

```toml
# in your plugin package's pyproject.toml
[project.entry-points."bdp_model_gate.checks"]
my_check = "my_package.checks:MyCustomCheck"
```

Once installed alongside `bdp-model-gate`, `default_structured_checks()`
picks it up automatically (pass `include_plugins=False` to opt out). A
plugin that fails to import or isn't a `BaseCheck` subclass is logged and
skipped rather than crashing the gate.

## Error handling

Bad inputs fail fast with a clear message rather than a confusing
exception from deep inside a check:

```python
from bdp_model_gate import ModelGate, StructuredGateContext
from bdp_model_gate.exceptions import GateValidationError

try:
    report = ModelGate().run(context)
except GateValidationError as exc:
    print(f"Fix your inputs: {exc}")
```

Validation covers: the model exposes `.predict()`, `X` is a non-empty
DataFrame, `y_true`/`y_pred`/`X` are aligned in length, `y_true` has at
least two classes, `protected_df` is row-aligned and has no all-NaN
columns, `model_card` is a dict, `generate_fn` is callable, and
`latencies_ms` has no negative values.

## Roadmap

- Unstructured data support (text/image/audio) — `bdp_model_gate.unstructured` reserves
  the shape (`UnstructuredGateContext`, a matching check suite) but raises
  `NotImplementedError` until it lands.
- HTML/Markdown report rendering alongside `to_json()`.

## Development

```bash
pip install -e ".[dev,structured]"

ruff check .              # lint
ruff format .             # format
mypy bdp_model_gate       # type check
pytest -q                 # test (85% coverage floor enforced)
```

`.pre-commit-config.yaml` runs ruff, mypy, and basic hygiene checks on
every commit — install with `pip install pre-commit && pre-commit install`.

CI (`.github/workflows/ci.yml`) runs lint, type-check, and the test suite
across Python 3.9–3.12 on every push/PR, plus a **core-install job** with
no `structured` extra — that job is what keeps the graceful-degradation
paths (`NOT_APPLICABLE` results, metric fallback) honest. Tests that need
a real estimator `importorskip` on scikit-learn rather than failing there.

The matrix covers the whole `requires-python` range. Note
`[tool.mypy] python_version` is pinned to 3.12 for numpy's stubs, so 3.9
compatibility is enforced by ruff's `target-version` and the 3.9 test job
rather than by the type checker.

This is all separate from `ci_examples/`, which are pre-deployment gates
for models *built by* consumers of this library, not for the library's own
code.

A runnable end-to-end walkthrough of everything above lives in
[`examples/bdp_model_gate_walkthrough.ipynb`](examples/bdp_model_gate_walkthrough.ipynb),
committed with outputs so it reads without being run.

See [`CHANGELOG.md`](https://github.com/vanjy-eng/model-gate/blob/main/CHANGELOG.md) for release history.
