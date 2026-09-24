#!/usr/bin/env python3
"""Materialize the full DomainNet-345 HF mirror into the native file layout.

The public ``wltjr1007/DomainNet`` mirror keeps the original relative image
paths and numeric 345-way labels in Parquet shards.  This utility consumes one
shard at a time, so a shard can be deleted immediately after ingestion.  It
also writes per-shard list fragments transactionally; ``--finalize`` combines
those fragments into the train/test lists consumed by DomainNetMTDA.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset parent directory")
    parser.add_argument("--parquet", help="One HF Parquet shard to ingest")
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.finalize == (args.parquet is not None):
        parser.error("supply exactly one of --parquet or --finalize")
    if args.parquet and not args.split:
        parser.error("--split is required with --parquet")
    return args


def domain_root(root: Path) -> Path:
    return root / "DomainNet"


def ingest(args: argparse.Namespace) -> None:
    root = domain_root(Path(args.root))
    list_root = root / "image_list"
    root.mkdir(parents=True, exist_ok=True)
    list_root.mkdir(parents=True, exist_ok=True)
    shard = Path(args.parquet).resolve()
    stem = shard.stem.replace(".", "_")
    tmp_paths = {domain: list_root / f".hf_{args.split}_{domain}_{stem}.tmp" for domain in DOMAINS}
    done_paths = {domain: path.with_suffix(".txt") for domain, path in tmp_paths.items()}
    if all(path.is_file() for path in done_paths.values()):
        print(f"Already ingested {shard.name}; all list fragments exist")
        return
    for path in tmp_paths.values():
        path.unlink(missing_ok=True)

    handles = {domain: path.open("w", encoding="utf-8") for domain, path in tmp_paths.items()}
    counts: Counter[str] = Counter()
    try:
        parquet = pq.ParquetFile(shard)
        columns = ["image", "label", "domain", "image_path"]
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
            payloads = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            domain_ids = batch.column("domain").to_pylist()
            image_paths = batch.column("image_path").to_pylist()
            for payload, label, domain_id, relative in zip(payloads, labels, domain_ids, image_paths):
                domain = DOMAINS[int(domain_id)]
                relative_path = Path(relative)
                if relative_path.parts[0] != domain:
                    raise ValueError(f"Domain/path mismatch: {domain} vs {relative}")
                if not 0 <= int(label) < 345:
                    raise ValueError(f"Invalid DomainNet label {label} for {relative}")
                raw = payload.get("bytes") if isinstance(payload, dict) else None
                if not raw:
                    raise ValueError(f"Missing encoded image bytes for {relative}")
                destination = root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_file():
                    temporary = destination.with_name(destination.name + ".hf_tmp")
                    temporary.write_bytes(raw)
                    os.replace(temporary, destination)
                handles[domain].write(f"{relative_path.as_posix()} {int(label)}\n")
                counts[domain] += 1
    except Exception:
        for handle in handles.values():
            handle.close()
        for path in tmp_paths.values():
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.close()
        for domain in DOMAINS:
            os.replace(tmp_paths[domain], done_paths[domain])
        print(f"Ingested {shard.name}: " + ", ".join(f"{d}={counts[d]}" for d in DOMAINS))


def finalize(args: argparse.Namespace) -> None:
    root = domain_root(Path(args.root))
    list_root = root / "image_list"
    if not list_root.is_dir():
        raise FileNotFoundError(list_root)
    for split in ("train", "test"):
        for domain in DOMAINS:
            fragments = sorted(list_root.glob(f".hf_{split}_{domain}_*.txt"))
            if not fragments:
                raise FileNotFoundError(f"No {split}/{domain} list fragments under {list_root}")
            output = list_root / f"{domain}_{split}.txt"
            temporary = output.with_suffix(".txt.tmp")
            total = 0
            with temporary.open("w", encoding="utf-8") as handle:
                for fragment in fragments:
                    text = fragment.read_text(encoding="utf-8")
                    handle.write(text)
                    total += text.count("\n")
            os.replace(temporary, output)
            print(f"Wrote {output.name}: {total} rows")


def main() -> None:
    args = parse_args()
    if args.finalize:
        finalize(args)
    else:
        ingest(args)


if __name__ == "__main__":
    main()
