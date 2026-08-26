# Extending

## Writing a check

Subclass `BaseCheck`, set the class attributes, implement `run(context)`,
return a list of `CheckResult`.

```python
from bdp_model_gate import BaseCheck, CheckResult


class FeatureDriftCheck(BaseCheck):
    """Flags validation features whose mean has drifted from training."""

    name = "feature_drift"
    category = "performance"  # fairness | performance | compliance | security
    blocking = False  # drift warrants a look, not a hard stop
    supported_tasks = ("binary", "multiclass", "regression")

    def __init__(self, reference, max_z: float = 3.0):
        self.reference = reference
        self.max_z = max_z

    def run(self, context):
        results = []
        for col in context.X.select_dtypes(include=["number"]).columns:
            if col not in self.reference:
                continue
            sd = self.reference[col].std()
            if sd == 0:
                continue
            z = abs(context.X[col].mean() - self.reference[col].mean()) / sd
            if z > self.max_z:
                results.append(
                    CheckResult(
                        self.name,
                        self.category,
                        "DRIFT_RISK",
                        detail=f"{col} mean shifted {z:.2f} sd from training",
                        blocking=self.blocking,
                        metadata={"feature": col, "z_score": round(float(z), 3)},
                    )
                )
        return results or [
            CheckResult(
                self.name,
                self.category,
                "OK",
                f"no feature drifted beyond {self.max_z} sd",
                self.blocking,
            )
        ]
```

Run it alongside the standard suite:

```python
from bdp_model_gate.structured import default_structured_checks

checks = default_structured_checks(config) + [FeatureDriftCheck(X_train)]
report = ModelGate(checks=checks).run(context)
```

### Two attributes worth thinking about

**`blocking`** is the important one. `True` fails the build; `False` routes to
human review. Ask whether a false positive on your check should stop a
deployment at 2am. If not, it is non-blocking.

**`supported_tasks`** declares what your check can answer. It defaults to
every task, so a check written before 0.3.0 keeps working — but if yours only
makes sense for one, say so and the gate will report `NOT_APPLICABLE`
elsewhere rather than letting it produce a meaningless number.

### A broken check is contained

`ModelGate` catches exceptions per check and converts them to a blocking
`CHECK_ERROR`, so one bad check cannot take down the suite — the others still
run and the pipeline still stops.

That means a `CHECK_ERROR` in a report is *always* a bug or an explicit
expectation, never noise.

## Plugins

A separate package can register checks without forking, via the
`bdp_model_gate.checks` entry-point group:

```toml
# in your plugin package's pyproject.toml
[project.entry-points."bdp_model_gate.checks"]
my_check = "my_package.checks:MyCustomCheck"
```

Once installed alongside `bdp-model-gate`, `default_structured_checks()` picks
it up automatically. Pass `include_plugins=False` to opt out.

Each entry point must resolve to a `BaseCheck` **subclass**, not an instance,
and is constructed with no arguments — so a plugin needing configuration
should read it from its own defaults or environment. A plugin that fails to
import, or does not resolve to a `BaseCheck`, is logged and skipped rather
than crashing the gate.

```python
from bdp_model_gate.registry import discover_plugin_checks

print(discover_plugin_checks())
```

## Contributing

```bash
pip install -e ".[dev,structured]"

ruff check .        # lint
ruff format .       # format
mypy bdp_model_gate # types
pytest -q           # tests, 85% coverage floor
```

`.pre-commit-config.yaml` runs the same hygiene on every commit. CI runs lint,
types and the test suite across Python 3.9–3.13, plus a core-install job with
no `structured` extra — that job is what keeps the graceful-degradation paths
honest.
