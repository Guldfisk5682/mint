#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


MANIFEST_NAME = "experiment_manifest.json"


def canonical_payload(args):
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "code_commit": code_commit,
        "data": str(Path(args.data).expanduser().resolve()),
        "dataset_config": str(Path(args.dataset_config).expanduser().resolve()),
        "dataset_config_sha256": hashlib.sha256(
            Path(args.dataset_config).read_bytes()
        ).hexdigest(),
        "effective_opts": args.effective_opts.strip(),
        "extra_opts": args.extra_opts.strip(),
        "method_tag": args.method_tag,
        "post_init_load_epoch": args.post_init_load_epoch,
        "post_init_method_tag": args.post_init_method_tag,
        "seed": args.seed,
        "score_manifest_sha256": (
            hashlib.sha256(Path(args.score_manifest).read_bytes()).hexdigest()
            if args.score_manifest else ""
        ),
        "source": args.source,
        "targets": args.targets,
        "trainer": args.trainer,
        "trainer_config": str(Path(args.trainer_config).expanduser().resolve()),
        "trainer_config_sha256": hashlib.sha256(
            Path(args.trainer_config).read_bytes()
        ).hexdigest(),
    }


def payload_signature(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prevent accidental checkpoint resume with a different experiment config."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-tag", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--score-manifest", default="")
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--trainer-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--extra-opts", default="")
    parser.add_argument("--effective-opts", default="")
    parser.add_argument("--post-init-method-tag", default="")
    parser.add_argument("--post-init-load-epoch", type=int, default=-1)
    parser.add_argument("--allow-legacy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise SystemExit("Commit experiment code and configs before launching a run")
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / MANIFEST_NAME
    payload = canonical_payload(args)
    manifest = {"signature": payload_signature(payload), "experiment": payload}

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise SystemExit(
                "Experiment manifest mismatch in "
                f"{output_dir}. Use a new METHOD_TAG/output directory; do not resume "
                "a checkpoint under a different configuration."
            )
        print(f"Experiment manifest matches: {manifest_path}")
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.allow_legacy:
            raise SystemExit(
                f"Non-empty legacy output directory has no {MANIFEST_NAME}: {output_dir}. "
                "Use a new METHOD_TAG, or set ALLOW_LEGACY_OUTPUT_DIR=1 only after "
                "manually verifying that the historical configuration matches."
            )
        print(f"WARNING: allowing legacy output directory without manifest: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote experiment manifest: {manifest_path}")


if __name__ == "__main__":
    main()
