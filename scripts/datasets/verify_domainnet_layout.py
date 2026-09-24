#!/usr/bin/env python3
"""Validate the exact DomainNet layout consumed by DomainNetMTDA."""

import argparse
from collections import defaultdict
from pathlib import Path


DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
SPLITS = ("train", "test")


def resolve_dataset_dir(root: Path) -> Path:
    root = root.expanduser().resolve()
    nested = root / "DomainNet"
    return nested if nested.is_dir() else root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset parent containing DomainNet/, or the DomainNet directory itself",
    )
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.root)
    list_dir = dataset_dir / "image_list"
    if not list_dir.is_dir():
        raise SystemExit(f"Missing image-list directory: {list_dir}")

    missing = []
    duplicate_paths = []
    seen_paths = set()
    labels = set()
    counts = defaultdict(int)

    for domain in DOMAINS:
        if not (dataset_dir / domain).is_dir():
            missing.append(str(dataset_dir / domain))
        for split in SPLITS:
            list_path = list_dir / f"{domain}_{split}.txt"
            if not list_path.is_file():
                missing.append(str(list_path))
                continue
            with list_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        relative_path, raw_label = line.rstrip().rsplit(maxsplit=1)
                        label = int(raw_label)
                    except ValueError as exc:
                        raise SystemExit(
                            f"Malformed entry at {list_path}:{line_number}: {line!r}"
                        ) from exc
                    normalized_path = relative_path.replace("\\", "/")
                    if normalized_path in seen_paths:
                        duplicate_paths.append(normalized_path)
                    seen_paths.add(normalized_path)
                    labels.add(label)
                    counts[(domain, split)] += 1
                    image_path = dataset_dir / normalized_path
                    if not image_path.is_file():
                        missing.append(str(image_path))

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:20])
        raise SystemExit(
            f"DomainNet validation failed: {len(missing)} missing paths. First entries:\n{preview}"
        )
    if duplicate_paths:
        preview = "\n".join(f"  {path}" for path in duplicate_paths[:20])
        raise SystemExit(
            f"DomainNet validation failed: {len(duplicate_paths)} duplicate list paths. First entries:\n{preview}"
        )
    if labels != set(range(345)):
        missing_labels = sorted(set(range(345)) - labels)
        extra_labels = sorted(labels - set(range(345)))
        raise SystemExit(
            "DomainNet labels must be exactly 0..344; "
            f"missing={missing_labels}, extra={extra_labels}"
        )

    print(f"DomainNet root: {dataset_dir}")
    for domain in DOMAINS:
        print(
            f"{domain:10s} train={counts[(domain, 'train')]:6d} "
            f"test={counts[(domain, 'test')]:6d}"
        )
    print(f"Validated {len(seen_paths)} unique listed images and 345 labels")


if __name__ == "__main__":
    main()
