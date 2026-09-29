"""Summarize a saved LongMemEval check without loading an encoder or dataset."""
import argparse
import json
import sys
from pathlib import Path

from longmemeval import summarize_checks, write_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report")
    p.add_argument("--types", nargs="+", help="combine these types from the saved validation cases")
    p.add_argument("--out", help="optional summary JSON file")
    args = p.parse_args()
    with Path(args.report).open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("experiment") != "longmemeval_relevance_diagnostic":
        p.error("expected a LongMemEval diagnostic report from the updated --check command")
    if report.get("schema_version", 1) < 2:
        print("This older report lacks oracle, cosine-distribution and session diagnostics. "
              "Rerun the updated check using the same embedding cache to obtain them.", file=sys.stderr)
    rows = [r for r in report["cases"] if not args.types or r["question_type"] in args.types]
    budgets = report["arguments"]["budgets"]
    summary = summarize_checks(rows, budgets)
    result = dict(source=str(args.report), types=args.types, summary=summary,
                  question_ids=[r["question_id"] for r in rows],
                  note="Uses the saved run's validation cases; no split reassignment or model inference.")
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.out:
        if Path(args.out).resolve() == Path(args.report).resolve():
            p.error("summary output must not overwrite its source report")
        write_report(args.out, result)


if __name__ == "__main__":
    main()
