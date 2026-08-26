#!/usr/bin/env python3
"""Summarise a mutmut run, and refuse to report a score it cannot support.

Two traps this exists to avoid, both of which caught the first version of the
CI job:

1. `mutmut run` can fail to import the package and do nothing at all. With
   `|| true` in the workflow, that reported green.
2. `mutmut results` lists **only survivors**. Counting statuses from it and
   dividing yields a 0% kill rate whatever the truth, because a killed mutant
   never appears in that output.

So the counts come from the progress line mutmut prints during the run, which
is the only place the full tally appears:

    410/3430  🎉 179  🫥 100  ⏰ 0  🤔 0  🙁 131  🔇 0  🧙 0

Usage:
    mutmut run 2>&1 | tee mutation.log
    python scripts/mutation_report.py mutation.log [--min-tested N]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# mutmut 3 renders each status as an emoji in its progress line.
STATUS_ICONS = {
    "🎉": "killed",
    "🙁": "survived",
    "⏰": "timeout",
    "🤔": "suspicious",
    "🫥": "not covered",
    "🔇": "skipped",
    "🧙": "check was skipped",
}
# Statuses that mean the mutant genuinely ran and the suite gave a verdict.
VERDICTS = {"killed", "survived", "timeout", "suspicious"}

PAIR = re.compile(r"(" + "|".join(map(re.escape, STATUS_ICONS)) + r")\s*(\d+)")
TOTAL = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


def parse(text: str) -> tuple[dict[str, int], int] | None:
    """Returns (counts, total_generated) from a mutmut run log, or None.

    mutmut redraws its progress line with carriage returns, so the whole run
    can arrive as a single line containing every intermediate tally. Matching
    "the last one" mixes groups across redraws — an earlier version of this
    script reported 410 processed alongside 3139 verdicts.

    The counts only ever increase during a run, so taking the maximum seen for
    each status is both correct and immune to how the terminal output is
    chunked.
    """
    counts: dict[str, int] = {}
    for icon, value in PAIR.findall(text):
        status = STATUS_ICONS[icon]
        counts[status] = max(counts.get(status, 0), int(value))
    if not counts:
        return None
    total = max((int(b) for _, b in TOTAL.findall(text)), default=0)
    return counts, total


def survivors() -> list[str]:
    """`mutmut results` lists survivors — useful for *where* to add assertions,
    just not for counting."""
    for command in ([sys.executable, "-m", "mutmut", "results"], ["mutmut", "results"]):
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode == 0:
            return [line.strip() for line in done.stdout.splitlines() if "bdp_model_gate" in line]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="output captured from `mutmut run`")
    parser.add_argument(
        "--min-tested",
        type=int,
        default=50,
        help="fail if fewer than this many mutants got a verdict (default 50)",
    )
    parser.add_argument(
        "--min-kill-rate",
        type=float,
        default=None,
        help="optionally fail below this kill rate, as a fraction",
    )
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"no such log: {args.log}", file=sys.stderr)
        return 1

    parsed = parse(args.log.read_text(errors="replace"))
    if parsed is None:
        print(
            "could not find a progress line in the log — `mutmut run` produced no\n"
            "tally, which usually means it never ran a mutant.",
            file=sys.stderr,
        )
        return 1

    counts, total = parsed
    tested = sum(n for status, n in counts.items() if status in VERDICTS)
    killed = counts.get("killed", 0)

    print("Mutation testing")
    print("=" * 52)
    print(f"  mutants generated  {total}")
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        marker = " " if status in VERDICTS else "*"
        print(f"  {marker} {status:20} {n:6}")
    print("  * did not run, so not evidence either way")

    if tested:
        rate = killed / tested
        print(f"\n  kill rate        {killed}/{tested} = {rate:.1%}")
    else:
        rate = 0.0

    by_module: dict[str, int] = {}
    for line in survivors():
        module = line.split(".x")[0].strip().rstrip(":")
        by_module[module] = by_module.get(module, 0) + 1
    if by_module:
        print("\n  survivors by module — where assertions are missing:")
        for module, n in sorted(by_module.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {n:5}  {module}")

    if tested < args.min_tested:
        print(
            f"\nFAIL: only {tested} mutant(s) got a verdict, expected at least "
            f"{args.min_tested}. The run did no useful work.",
            file=sys.stderr,
        )
        return 1

    if args.min_kill_rate is not None and rate < args.min_kill_rate:
        print(
            f"\nFAIL: kill rate {rate:.1%} is below the floor of {args.min_kill_rate:.1%}.",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK: {tested} mutant(s) got a real verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
