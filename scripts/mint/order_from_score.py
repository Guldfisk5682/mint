#!/usr/bin/env python3
"""Read a label-free target-train difficulty manifest for an SS-MTDA run."""

import argparse
import json
from pathlib import Path

from score_utils import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument(
        "--direction", choices=("hard_to_easy", "easy_to_hard"), default="hard_to_easy"
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if payload.get("source") != args.source or payload.get("seed") != args.seed:
        raise ValueError("Difficulty manifest source/seed differs from this run")
    if payload.get("scoring_split") != "target_train":
        raise ValueError("Difficulty must be scored on unlabeled target-train images")
    if payload.get("selection_uses_target_labels") is not False:
        raise ValueError("Target labels must not select the curriculum order")
    if payload.get("selection_uses_target_test_inputs") is not False:
        raise ValueError("Target-test inputs must not select the curriculum order")
    if any("accuracy" in metrics for metrics in payload["scores"].values()):
        raise ValueError("Difficulty manifest unexpectedly contains target accuracy")
    checkpoint = (
        Path(payload["model_dir"]) / "ContinuousSharedProjMaPLeMTDA"
        / f"model.pth.tar-{payload['load_epoch']}"
    )
    actual_hash = sha256_file(checkpoint)
    if actual_hash != payload.get("checkpoint_sha256"):
        raise ValueError("Difficulty score does not match its source-only checkpoint")
    order = payload[args.direction]
    if len(order) != len(args.targets) or set(order) != set(args.targets):
        raise ValueError("Difficulty order does not match the requested target domains")
    print(*order, sep="\n")


if __name__ == "__main__":
    main()
