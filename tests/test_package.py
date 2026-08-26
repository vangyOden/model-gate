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
