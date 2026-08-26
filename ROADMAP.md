# Roadmap

Planned work, with enough detail that decisions already taken do not get
re-argued. Shipped releases are in [`CHANGELOG.md`](CHANGELOG.md).

Current release: **0.4.2**.

---

## 0.4.1-alpha — Documentation site

Tracked separately from the library, which is already at 0.4.1 on PyPI. The
`-alpha` marks the **site**, not the package: it lives in `web/`, is not yet
deployed, and its structure will move before it is announced.

### Why

`README.md` reached 604 lines and 15 top-level sections. Someone who wants
regression scrolls past binary classification, metric selection, custom
checks and plugins to reach it. Splitting that up is the actual win; the site
is the means.

### Audience

External — banks, insurers and any organisation with a working data-science
team. Two consequences:

- The generic quickstart leads. NDPA/NDPR defaults are presented as
  **configurable defaults**, not the product's premise, so a reader in
  another regime sees themselves in the hero.
- Insurance use cases are the worked examples rather than the framing.

### Shape: landing + docs, as pandas does it

Two builds under one deploy, mirroring `pandas.pydata.org` (a hand-built
marketing root, with Sphinx docs beneath it):

```
web/
  landing/index.html    hand-built landing page, deployed at /
  mkdocs.yml            MkDocs Material, deployed at /docs/
  docs/                 the guide, reference and rendered notebooks
  requirements.txt      pinned docs toolchain
```

- **MkDocs Material**, not Sphinx: the content is already Markdown, and the
  API is 53 public objects — not the scale where intersphinx and autodoc
  earn their configuration cost.
- **`mkdocstrings`** generates the API reference from the docstrings that
  already cover 89% of the public API, so it cannot drift from the source.
- **`mkdocs-jupyter`** renders the five executed notebooks as pages, so
  `examples/run_all.sh` keeps them honest and there is no second copy.
- Multi-version docs (`mike`) deferred to 1.0 — pre-1.0 and moving this
  fast, one accurate "latest" beats five stale versions.

### Known risk

A site multiplies the surface that can go stale, and this project has form:
a notebook shipped two minor versions behind, and twice a notebook's prose
contradicted its own output. The generated API reference and rendered
notebooks are structurally protected. **Prose code blocks are not** —
executing them in CI is the open question, deferred to the 0.4.2 robustness
work rather than decided here.

---

## 0.4.2 — Robustness of the checks themselves ✅

**Shipped.** See [`CHANGELOG.md`](CHANGELOG.md) for detail. The suite went from
167 to 256 tests across five new files: known-answer, metamorphic invariant,
model-family matrix, property-based, and skip-reason coverage — plus an autouse
`CHECK_ERROR` guard and advisory mutation testing.

Two more bugs surfaced while building it, both found by the new tests rather
than by inspection: `shap_gap_threshold` was absolute where it needed to be
relative, and subsampling selected rows by position, so sorting a CSV could
change a verdict.

Still open from the original plan:

- **A mutation kill-rate floor.** The CI job is advisory until the score is
  known and stable; turning it into a threshold is the follow-up.

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
