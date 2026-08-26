# Roadmap

Planned work, with enough detail that decisions already taken do not get
re-argued. Shipped releases are in [`CHANGELOG.md`](CHANGELOG.md).

Current release: **0.4.1**.

---

## 0.4.2 — Robustness of the checks themselves

Five silent failures have shipped and been fixed so far. Every one passed the
test suite at the time, and for a tool whose job is catching problems that is
the pattern worth attacking directly. Each had a specific structural cause,
and that mapping is the plan:

| Bug that shipped | Why tests missed it | Guard to build |
|---|---|---|
| `DisparateImpactCheck` returned `0.000` for a maximally unfair model | no test knew the *correct answer* | known-answer tests |
| `ShapSubgroupCheck` crashed on `RandomForestClassifier` | one model family in fixtures; `assert len(results) > 0` passed on a `CHECK_ERROR` | model-family matrix + global `CHECK_ERROR` guard |
| CLI sliced `predict_proba[:, 1]` on multiclass | CLI tested only for binary | cross-task CLI matrix |
| adversarial perturbation scaled off the largest column | all fixture features had similar magnitude | scale-invariance test + heterogeneous fixture |
| gradient attack was weaker than random noise | no test compared the two | already added in 0.4.1 — keep as the template |

### Activities, in value order

1. **Autouse `CHECK_ERROR` guard** in `conftest.py`. Any test whose report
   contains a `CHECK_ERROR` fails unless it opts in via a marker. A
   `CHECK_ERROR` is *always* either a bug or an explicit expectation. This
   one fixture would have caught the SHAP crash outright.
2. **Known-answer suite** — one per check, with hand-derived values: parity
   difference exactly `1.0`/`0.0`, η² exactly `1.0` for a feature that is a
   pure function of the attribute, hand-computed `ordinal_mae`,
   `quadratic_kappa` and loss-ratio figures, flip rate exactly `0` for a
   constant model. If you know the answer, wrong cannot hide.
3. **Metamorphic invariants** — properties that must hold for *any* input,
   which no example test covers:
   - row-permutation invariance
   - class-relabelling invariance for ordinal metrics
   - **feature-scale invariance** (would have caught the perturbation bug)
   - `y`-scaling: `rmse`/`mae` scale by *k*, `r2` unchanged
   - group-relabelling invariance
   - monotonicity — making a model strictly more unfair must never lower the
     disparity figure
4. **Model-family × task matrix** — one parametrized suite over
   `LogisticRegression`, `RandomForest`, `GradientBoosting`,
   `SVC(probability=True)`, a `Pipeline` and a `predict_fn` closure, crossed
   with binary / multiclass / regression. Assert every flag lands in a known
   set.
5. **Mutation testing** — `mutmut` over `bdp_model_gate/`, reported as a
   non-blocking CI job first, then a kill-rate floor. The only technique that
   actually answers *"would my tests have noticed?"*, which is the question
   this whole release exists to answer.
6. **Property-based layer** — `hypothesis` on the metric functions and
   `ModelAdapter` shape handling (bounds, direction, symmetry).
7. **`NOT_APPLICABLE` reason coverage** — every skip path asserted on its
   reason string. Those paths are where a check silently does nothing, which
   is the failure mode that hides best.
8. **Hostile fixtures** — single-row groups, constant features, 99.9/0.1
   imbalance, zero-variance target, all-NaN optional inputs.

### Also in 0.4.2

- **Make `FairnessConfig.shap_gap_threshold` relative.** It is currently
  absolute and expressed in the units of the model output, unlike the four
  regression fairness thresholds. On a premium model predicting naira in the
  tens of thousands the default `0.15` flags essentially every feature — 12
  of 17 fairness findings in the regression example before it was rescaled by
  hand. A check that flags everything is as useless as one that flags
  nothing. Notebook `03` documents the workaround meanwhile.

---

## 0.4.3 — Tooling and CI pinning

Unpinned linters change their verdict on unchanged code, which produces a
confusing red build months later on an unrelated PR.

1. Pin `ruff` and `mypy` **exactly** (`==`) in a dedicated `lint` extra,
   separate from `dev`, so lint and test dependencies move independently.
2. **Reconcile pre-commit with CI.** `.pre-commit-config.yaml` pins ruff
   `v0.13.2` while CI installs the latest (`0.16.4` at time of writing) — so
   a developer running pre-commit and CI can disagree *today*. The rev should
   be derived from the pinned version rather than maintained by hand.
3. **Scheduled "latest tooling" job** — weekly, non-blocking, running
   unpinned `ruff`/`mypy` so upgrades surface as a deliberate decision rather
   than a surprise failure.
4. Bump `actions/*` past the Node 20 deprecation warnings now appearing in
   every run.
5. A constraints file, so a lint run is byte-reproducible.

---

## 0.4.4 — Release automation

Publishing is currently manual. Two of the five silent failures were caught
*only* by installing the published artifact, and 0.3.0 and 0.3.1 were both
tagged on commits with a red Python 3.9 job — so the release path itself is
worth gating. Fitting, for a tool that exists to gate deploys.

### Design decisions already made

- **Trigger on the tag, not on merge to `main`.** Not every merge is a
  release, and tags already mark them. `on: push: tags: ['v*']` means the
  artifact published is the commit that was tagged, with no drift between
  "what merged" and "what shipped".
- **Trusted Publishing (OIDC), not API tokens.** `pypa/gh-action-pypi-publish`
  with `id-token: write`. No long-lived credential in repo secrets, nothing
  to rotate or leak.
- **Protection belongs on a GitHub Environment, not the branch.** Branch
  protection guards what gets *merged*; an Environment with a required
  reviewer guards what gets *published*. Both are wanted, but only the
  Environment stands between a tag and PyPI.

### Pipeline

```
tag v* pushed
  ├─ build        sdist + wheel, then run the test suite against the
  │               installed artifact rather than the source tree
  ├─ testpypi     environment: testpypi — no approval
  ├─ smoke-test   fresh venv, install from TestPyPI, import, run a real gate
  └─ pypi         environment: pypi — REQUIRED REVIEWER
```

### Guards to build in

- **Tag/version consistency** — fail if the tag does not match
  `pyproject.toml`. `tests/test_package.py` already ties the version to
  `__version__` and the changelog; this closes the last gap.
- **Publish only on a green matrix for that commit.** Both 0.3.0 and 0.3.1
  were tagged with a failing 3.9 job. A release workflow that re-runs the
  matrix makes that structurally impossible.
- **Smoke-test from TestPyPI before PyPI.** The 3.9 import failure lived in
  the *published wheel*, and the shap/numpy incompatibility only appeared on
  a clean install. Neither was visible in the source tree.

### Manual prerequisites (repo admin, not code)

These cannot be done from a workflow and need doing before the first run:

- [ ] Register the repository, workflow filename and environment name as a
      **trusted publisher on PyPI**
- [ ] The same on **TestPyPI** — a separate registration
- [ ] Create the `testpypi` and `pypi` **Environments**, with a required
      reviewer on `pypi`
- [ ] Enable **branch protection** on `main`: require a PR and a green CI run

Two things to know going in:

- If the workflow **filename** later changes, publishing breaks until the
  trusted-publisher config is updated to match.
- **Version collisions are permanent on both indexes.** Neither allows
  re-uploading a version, and `0.2.0`, `0.2.1`, `0.3.2`, `0.4.0` and `0.4.1`
  are already taken. A failed release means bumping the patch, not retrying.

---

## Later

- **A public, subclassable `ModelAdapter` (1.0.0).** The extension point is a
  plain callable for now, which covers every case with less ceremony; a named
  class earns its place once someone needs to attach batching, retries or
  auth to a serving layer.
- **Unstructured data support** (text / image / audio).
  `bdp_model_gate.unstructured` reserves the shape —
  `UnstructuredGateContext` and a matching check suite — and raises
  `NotImplementedError` until it lands.
- **HTML/Markdown report rendering** alongside `to_json()`.
