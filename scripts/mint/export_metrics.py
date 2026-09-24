#!/usr/bin/env python3
"""Export final per-target accuracy and executed stage order from a run."""

import argparse
import json
import re
from pathlib import Path


TARGET_RE = re.compile(r"Target domain ([A-Za-z_]+) accuracy: ([0-9]+(?:\.[0-9]+)?)%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    args = parser.parse_args()

    log_path = args.run_dir / "log.txt"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Deploy the last-epoch model" not in text:
        raise ValueError("Run lacks evidence of last-epoch checkpoint evaluation")
    found = TARGET_RE.findall(text)
    accuracies = {}
    for domain, score in found:
        accuracies[domain] = float(score)
    if set(accuracies) != set(args.targets):
        raise ValueError(f"Expected {args.targets}, found {accuracies}")

    audit_path = args.run_dir / "curriculum_stage_audit.jsonl"
    audits = []
    if audit_path.exists():
        audits = [json.loads(line) for line in audit_path.read_text().splitlines() if line]
    payload = {
        "dataset": args.dataset,
        "method": args.method,
        "source": args.source,
        "seed": args.seed,
        "per_domain_accuracy": accuracies,
        "macro_accuracy": sum(accuracies.values()) / len(accuracies),
        "executed_stage_order": [row["domain"] for row in audits],
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in audits),
        "checkpoint_selection": "last_step",
    }
    output = args.run_dir / "mtda_metrics.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
