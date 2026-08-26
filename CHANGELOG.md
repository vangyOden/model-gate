# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.1] - 2026-08-26

### Fixed
- **`DisparateImpactCheck` silently reported `OK` for probability
  predictions.** Demographic parity compares selection rates, counting
  predictions equal to `1`. A probability is never exactly `1`, so every
  group's selection rate came out as `0` and the parity difference was
  always exactly `0.000` — the check passed no matter how skewed the model
  was. Verified against a maximally discriminatory model (every man
  selected, no women): the check reported `0.000` where the true parity
  difference is `1.000`. This mattered in practice because the documented
  quickstart passes `model.predict_proba(X)[:, 1]`, so the default usage
  disabled the check. `y_pred` is now binarised at the new
  `FairnessConfig.decision_threshold` (default `0.5`); predictions already
  in `{0, 1}` are untouched, so callers passing hard labels see no change.
- **`ShapSubgroupCheck` crashed on `RandomForestClassifier`.** shap returns
  a 3-D `(rows, features, classes)` array for some classifiers and 2-D for
  others, and which you get changed across shap versions. Building a
  DataFrame from the 3-D form raised `ValueError: Must pass 2-d input`,
  which the gate surfaced as a blocking `CHECK_ERROR`. Binary output is now
  reduced to the positive class; genuine multiclass reports
  `NOT_APPLICABLE` rather than guessing at a class.
- **The `structured` extra resolved to a combination that could not
  import.** `shap>=0.44,<0.47` is incompatible with `numpy>=2.0`, which the
  `numpy>=1.23,<3.0` range permits, so a fresh install produced shap 0.46
  alongside numpy 2.x and `import shap` raised `TypeError: Converting
  'np.inexact' or 'np.floating' to a dtype not allowed`. Since
  `ShapSubgroupCheck` catches only `ImportError`, this surfaced as a
  blocking `CHECK_ERROR` rather than a graceful skip. The floor is now
  `shap>=0.48` (imports cleanly against numpy 2.x, ships cp313 wheels) and
  the ceiling `<0.50`, which raised its own Python floor to 3.11 — above
  this package's `requires-python`.

### Added
- `FairnessConfig.decision_threshold` — the cutoff used to binarise
  continuous predictions for `DisparateImpactCheck`. Reported in that
  check's result metadata so a report states what it measured against.
- `examples/bdp_model_gate_walkthrough.ipynb` — an end-to-end notebook
  covering the full public API, committed with outputs.
- `tests/test_fairness_fixes.py` — regression tests for both fairness bugs,
  including one that asserts on the result *flag* rather than merely that
  results exist. Asserting existence is why the SHAP crash went unnoticed:
  routed through `ModelGate`, a raising check still produces a result.

### Changed
- CI test matrix extended to Python 3.13, now that `shap>=0.48` supports it.


## [0.2.0] - 2026-08-26

### Added
- **Configurable performance metric.** `PerformanceConfig.metric` selects
  what the model is scored on: a built-in name (`roc_auc`,
  `average_precision`, `accuracy`, `balanced_accuracy`, `f1`, `precision`,
  `recall`), a `fn(y_true, y_pred) -> float` callable, or `"auto"`. A new
  `bdp_model_gate.metrics` module owns resolution.
- `PerformanceConfig.decision_threshold` (default `0.5`) — binarizes
  continuous predictions for metrics that need hard class labels.
  Predictions already in `{0, 1}` are untouched; ranking metrics ignore it.
- `GateReport.model_metric` / `model_score`, and a `model_metric` /
  `model_score` pair in `to_dict()` / `to_json()`. `summary()` now prints
  the headline score.
- CLI `--metric`, `--min-score`, and `--decision-threshold` flags, which
  take precedence over `--config` file values.
- `AdversarialRobustnessCheck(random_state=...)` to control its sampling
  and perturbation seed.
- `ModelGate` now dispatches input validation on `context.modality` via a
  `VALIDATORS` registry, instead of hardcoding the structured validator —
  an unknown modality raises `GateValidationError` rather than being
  silently validated as structured.
- `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, and `.gitignore`,
  which the README described but the repo didn't contain. CI gained a
  core-install job (no `structured` extra) that exercises the
  graceful-degradation and metric-fallback paths, and a build job.
- Tests: `tests/test_metrics.py` covering metric selection, fallback
  visibility, thresholding and the deprecation aliases; plus tests for
  adversarial determinism, coefficient alignment, and modality dispatch.
- Eager input validation (`GateValidationError`) — bad inputs now fail
  fast with a clear message instead of an opaque exception from inside a check.
- Structured `logging` throughout the library and CLI, replacing `print()`.
  CLI gained a `-v/--verbose` flag.
- Per-check timing (`CheckResult.duration_ms`, `GateReport.total_duration_ms`).
- Plugin system: third-party checks can be registered via the
  `bdp_model_gate.checks` entry-point group and are picked up automatically
  by `default_structured_checks()`.
- CLI `--config` now accepts YAML and TOML in addition to JSON.
- `ShapSubgroupCheck` now uses `shap.TreeExplainer` automatically for
  tree-based models (much faster and exact, vs. the generic explainer).
- `AdversarialRobustnessCheck` now uses a gradient-directed perturbation
  for linear models (via `model.coef_`) instead of always using isotropic
  random noise; falls back to random perturbation for black-box models.
- `py.typed` marker (PEP 561) — type checkers now see this package's
  annotations.
- Full type-hint pass across the codebase; `mypy` and `ruff` configured
  and added to `dev` extras.
- Expanded test suite: input-validation tests, edge cases (models without
  `predict_proba`, all-categorical features, a check that raises), plugin
  registry test, and `pytest-cov` with an 85% coverage floor enforced in CI.
- `tests/conftest.py` with shared fixtures (previously duplicated per test file).
- CI workflow (`.github/workflows/ci.yml`) running lint, type-check, and
  tests on every push/PR — separate from the pre-deployment gate workflows.

### Changed
- Renamed package from `mlgate` to `bdp_model_gate` (distribution name
  `bdp-model-gate`, CLI command `bdp-model-gate`).
- Dependency version ranges are now upper-bounded, not just floored.

### Deprecated
- `PerformanceConfig.min_accuracy` → use `min_score` and set `metric` to
  name what it applies to. The old name still works, in Python and in
  `--config` files, but emits a `DeprecationWarning`; the CLI additionally
  logs a rename notice.
- `GateReport.model_auc` → use `model_metric` + `model_score`. It now
  returns `None` unless the configured metric really was `roc_auc`, rather
  than mislabelling another metric's score as an AUC. The `model_auc` key
  remains in the JSON report for existing consumers, under the same rule.

### Fixed
- **`AdversarialRobustnessCheck` was not deterministic.** Its random
  perturbation used unseeded `np.random`, so the same model and data could
  produce a different flip rate — and a different gate verdict — between
  runs. Now seeded via `random_state` (default 42), matching the sampling
  already done elsewhere in the check.
- **`AdversarialRobustnessCheck` misaligned linear coefficients.**
  `coef_` is laid out over every column the model was fitted on, but was
  indexed by position among the *numeric* columns — so any non-numeric
  column ahead of a numeric one applied the wrong feature's weight. Now
  indexed by position in `X.columns`, with an explicit bail-out to random
  perturbation when the lengths don't line up. Multiclass `coef_` (one row
  per class) also falls back, rather than being flattened into a
  meaningless direction.
- **The performance gate silently changed metrics.** `min_accuracy` was
  compared against ROC AUC when scikit-learn was installed and against
  accuracy when it wasn't, with nothing in the report saying which. The
  metric is now explicit, and any fallback is logged at `WARNING`, flagged
  in the result metadata, and named in the detail string.
- `GateReport.model_auc` was populated by recomputing ROC AUC in the gate,
  independently of what the performance check actually gated on.
- Docstrings still referring to the pre-rename `mlgate` package name.
- Removed the empty, unused `bdp_model_gate/ci_examples/` directory (the
  real examples live in the top-level `ci_examples/`).

## [0.1.0] - 2026-08-22

### Added
- Initial release: fairness (proxy correlation, disparate impact, SHAP
  subgroup gaps, counterfactual flip), performance thresholds, NDPA/NDPR
  compliance mapping, and security checks (adversarial robustness, PII
  leakage, prompt injection).
- `ModelGate` / `GateReport` / `StructuredGateContext` core API.
- `bdp-model-gate` CLI for CI/CD use.
- Azure Pipelines and GitHub Actions pre-deployment gate examples.

[Unreleased]: https://github.com/vanjy-eng/model-gate/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/vanjy-eng/model-gate/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/vanjy-eng/model-gate/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vanjy-eng/model-gate/releases/tag/v0.1.0
