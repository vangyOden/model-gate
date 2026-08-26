"""Package-level invariants: the version is declared in two places and a
release tag is cut against both, so they must not drift apart."""

import re
from pathlib import Path

import bdp_model_gate

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    """Read the version straight out of pyproject.toml rather than via
    importlib.metadata — an editable install caches its metadata at install
    time, so a fresh bump would compare against a stale value and pass."""
    match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(), re.M)
    assert match, "no version declared in pyproject.toml"
    return match.group(1)


def test_dunder_version_matches_pyproject():
    assert bdp_model_gate.__version__ == _pyproject_version()


def test_changelog_documents_the_current_version():
    changelog = (PYPROJECT.parent / "CHANGELOG.md").read_text()
    assert f"## [{_pyproject_version()}]" in changelog


def test_no_runtime_pep604_without_future_import():
    """Guards the bug that took down the whole Python 3.9 CI job.

    `X | None` in an annotation is evaluated at runtime unless the module has
    `from __future__ import annotations`. On 3.10+ that is fine, so the
    problem is invisible on a modern interpreter — but on 3.9 it is an
    import-time TypeError, and a dataclass field annotation makes it fire on
    import, taking every test module down at collection.

    Ruff's FA102 also catches this; this test means the guard survives even
    if the lint config changes.
    """
    import ast

    offenders = []
    for path in sorted((PYPROJECT.parent / "bdp_model_gate").rglob("*.py")):
        tree = ast.parse(path.read_text())
        has_future = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body
        )
        if has_future:
            continue

        annotations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                annotations.append(node.annotation)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotations.extend(a.annotation for a in node.args.args if a.annotation)
                if node.returns:
                    annotations.append(node.returns)

        if any(
            isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr)
            for annotation in annotations
            for inner in ast.walk(annotation)
        ):
            offenders.append(path.name)

    assert not offenders, (
        f"{offenders} use PEP 604 unions without `from __future__ import annotations`, "
        "which is an import-time TypeError on Python 3.9"
    )
