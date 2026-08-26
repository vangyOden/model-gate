#!/usr/bin/env bash
# Re-execute every example notebook in place and fail on the first error.
#
# The notebooks are committed with outputs, so those outputs are a claim about
# how the library behaves. Run this whenever that behaviour changes: stale
# outputs are worse than none, because they look authoritative.
#
#   ./examples/run_all.sh            # all notebooks
#   ./examples/run_all.sh 03 04      # only those matching a prefix
set -euo pipefail

cd "$(dirname "$0")"

if ! python -c "import nbconvert" 2>/dev/null; then
    echo "nbconvert is not installed. Try: pip install jupyter" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    notebooks=()
    for prefix in "$@"; do
        while IFS= read -r match; do notebooks+=("$match"); done < <(ls "${prefix}"*.ipynb)
    done
else
    notebooks=(*.ipynb)
fi

failed=()
for notebook in "${notebooks[@]}"; do
    echo "==> $notebook"
    if python -m nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1200 "$notebook" >/dev/null 2>&1; then
        # nbconvert exits non-zero on an error, but check the saved outputs
        # too — a cell can record an error without failing the run.
        if python - "$notebook" <<'PY'
import json, sys
nb = json.load(open(sys.argv[1]))
bad = [
    (i, o["ename"], o["evalue"])
    for i, c in enumerate(nb["cells"])
    if c["cell_type"] == "code"
    for o in c.get("outputs", [])
    if o.get("output_type") == "error"
]
for i, name, value in bad:
    print(f"    cell {i}: {name}: {value}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
        then
            echo "    ok"
        else
            failed+=("$notebook")
        fi
    else
        echo "    FAILED to execute" >&2
        failed+=("$notebook")
    fi
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo >&2
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
echo
echo "all ${#notebooks[@]} notebook(s) executed cleanly"
