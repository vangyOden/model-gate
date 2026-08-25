"""Command-line entry point for running the gate in a CI/CD pipeline.

Installed as the `bdp-model-gate` console script. Exit codes are chosen so
a pipeline can distinguish "safe to proceed", "needs a human", and "hard
stop":

    0 -> PASS           safe to proceed automatically
    2 -> NEEDS_REVIEW    route to a manual approval step, don't auto-deploy
    1 -> BLOCKED         hard fail the pipeline

Example (Azure Pipelines / GitHub Actions):

    bdp-model-gate \
        --model model.joblib \
        --data validation.csv \
        --target-col label \
        --protected protected.csv \
        --model-card model_card.json \
        --cost-per-inference 0.0008 \
        --output gate_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ._logging import configure_logging, get_logger
from .exceptions import BDPModelGateError
from .metrics import AUTO, BUILTIN_METRICS

logger = get_logger("cli")

#: Config-file keys that have been renamed. Still applied (via the property
#: alias on the config dataclass), but called out in the log — a silently
#: honoured deprecated key is how a stale threshold survives a rename.
DEPRECATED_CONFIG_KEYS = {
    ("performance", "min_accuracy"): "min_score",
}


def _load_model(path: str):
    import joblib

    return joblib.load(path)


def _predict(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _load_structured_config_file(path: str) -> dict[str, Any]:
    """Loads threshold overrides from JSON, YAML, or TOML, based on extension."""
    suffix = Path(path).suffix.lower()
    text = Path(path).read_text()

    if suffix == ".json":
        return json.loads(text)

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise BDPModelGateError(
                "reading a YAML config requires PyYAML — install with `pip install pyyaml`"
            ) from exc
        return yaml.safe_load(text) or {}

    if suffix == ".toml":
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]  # backport for < 3.11
            except ImportError as exc:
                raise BDPModelGateError(
                    "reading a TOML config on Python < 3.11 requires tomli — "
                    "install with `pip install tomli`"
                ) from exc
        return tomllib.loads(text)

    raise BDPModelGateError(
        f"unrecognized config file extension '{suffix}' — use .json, .yaml/.yml, or .toml"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bdp-model-gate",
        description="Run the BDP Model Gate pre-deployment governance gate against a trained model.",
    )
    parser.add_argument("--model", required=True, help="Path to a joblib-serialized model")
    parser.add_argument("--data", required=True, help="Path to a CSV of validation data")
    parser.add_argument("--target-col", required=True, help="Column name of the ground-truth label")
    parser.add_argument(
        "--protected", help="Path to a CSV of protected attributes, row-aligned to --data"
    )
    parser.add_argument("--model-card", help="Path to a JSON model card")
    parser.add_argument(
        "--latencies", help="Path to a text/CSV file of per-request latencies in ms, one per line"
    )
    parser.add_argument("--cost-per-inference", type=float, help="Estimated cost per inference")
    parser.add_argument(
        "--metric",
        choices=[AUTO, *sorted(BUILTIN_METRICS)],
        help=(
            "Metric the model is scored on for the performance gate "
            f"(default: {AUTO}, which prefers roc_auc and falls back to accuracy "
            "with a warning if scikit-learn is unavailable)"
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum acceptable value of --metric; below this the gate blocks",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        help=(
            "Probability cutoff used to turn continuous predictions into class "
            "labels for metrics that need them (accuracy, f1, precision, recall). "
            "Ignored by ranking metrics like roc_auc."
        ),
    )
    parser.add_argument(
        "--config", help="Path to a JSON, YAML, or TOML file of threshold overrides"
    )
    parser.add_argument(
        "--output", default="gate_report.json", help="Where to write the JSON report"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug-level logging")
    return parser


def _apply_config_overrides(gate_config, overrides: dict[str, Any]):
    for section, values in overrides.items():
        sub_config = getattr(gate_config, section, None)
        if sub_config is None:
            logger.warning("config file references unknown section '%s' — ignoring", section)
            continue
        for key, value in values.items():
            replacement = DEPRECATED_CONFIG_KEYS.get((section, key))
            if replacement is not None:
                logger.warning(
                    "config key '%s.%s' is deprecated — rename it to '%s.%s'. Applying it for now.",
                    section,
                    key,
                    section,
                    replacement,
                )
            elif not hasattr(sub_config, key):
                logger.warning("config section '%s' has no field '%s' — ignoring", section, key)
                continue
            setattr(sub_config, key, value)
    return gate_config


def _apply_cli_overrides(gate_config, args):
    """CLI flags win over the --config file, so a pipeline can pin a
    threshold inline without maintaining a separate config file."""
    for flag_name, config_field in (
        ("metric", "metric"),
        ("min_score", "min_score"),
        ("decision_threshold", "decision_threshold"),
    ):
        value = getattr(args, flag_name, None)
        if value is not None:
            setattr(gate_config.performance, config_field, value)
            logger.debug("performance.%s set to %r from --%s", config_field, value, flag_name)
    return gate_config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        from bdp_model_gate import GateConfig, ModelGate, StructuredGateContext
        from bdp_model_gate.exceptions import GateValidationError
        from bdp_model_gate.structured import default_structured_checks

        model = _load_model(args.model)
        df = pd.read_csv(args.data)
        y_true = df[args.target_col].values
        X = df.drop(columns=[args.target_col])
        y_pred = _predict(model, X)

        protected_df = pd.read_csv(args.protected) if args.protected else None
        model_card = json.load(open(args.model_card)) if args.model_card else None

        latencies_ms = None
        if args.latencies:
            with open(args.latencies) as f:
                latencies_ms = [float(line.strip()) for line in f if line.strip()]

        gate_config = GateConfig()
        if args.config:
            overrides = _load_structured_config_file(args.config)
            gate_config = _apply_config_overrides(gate_config, overrides)
        gate_config = _apply_cli_overrides(gate_config, args)

        context = StructuredGateContext(
            model=model,
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            protected_df=protected_df,
            latencies_ms=latencies_ms,
            cost_per_inference=args.cost_per_inference,
            model_card=model_card,
        )

        report = ModelGate(checks=default_structured_checks(gate_config)).run(context)

    except GateValidationError as exc:
        logger.error("invalid gate input: %s", exc)
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1
    except BDPModelGateError as exc:
        logger.error("configuration error: %s", exc)
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        logger.error("file not found: %s", exc)
        print(f"File not found: {exc}", file=sys.stderr)
        return 1

    report.to_json(args.output)
    print(report.summary())
    print(f"Full report written to {args.output}")

    if report.gate_status == "BLOCKED":
        return 1
    if report.gate_status == "NEEDS_REVIEW":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
