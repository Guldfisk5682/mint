#!/usr/bin/env python3

import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
BLOCKED_PATHS = {
    REPO_ROOT,
    REPO_ROOT / "datasets",
    SCRIPT_PATH.parent,
}

clean_sys_path = []
for entry in sys.path:
    entry_path = Path(entry or os.getcwd()).resolve()
    if entry_path in BLOCKED_PATHS:
        continue
    clean_sys_path.append(entry)
sys.path = clean_sys_path

from datasets import Image, load_dataset


DOMAIN_MAP = {
    "art": "art",
    "clipart": "clipart",
    "product": "product",
    "real world": "real_world",
    "real_world": "real_world",
    "real-world": "real_world",
}


def normalize_domain_name(domain_name: str) -> str:
    key = domain_name.strip().lower().replace("-", " ").replace("_", " ")
    if key not in DOMAIN_MAP:
        raise ValueError(f"Unsupported Office-Home domain from HF dataset: {domain_name}")
    return DOMAIN_MAP[key]


def normalize_class_name(class_name: str) -> str:
    return (
        class_name.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def main():
    repo_id = os.environ.get("HF_DATASET_REPO", "flwrlabs/office-home")
    split = os.environ.get("HF_DATASET_SPLIT", "train")
    target_dir = Path(os.environ["TARGET_DIR"]).resolve()
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    local_parquet_dir = os.environ.get("HF_LOCAL_PARQUET_DIR")

    os.environ["HF_ENDPOINT"] = hf_endpoint
    if local_parquet_dir:
        parquet_paths = sorted(Path(local_parquet_dir).glob("*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(
                f"No local parquet shards found under HF_LOCAL_PARQUET_DIR={local_parquet_dir}"
            )
        dataset = load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in parquet_paths]},
            split=split,
        )
    else:
        dataset = load_dataset(repo_id, split=split)

    label_feature = dataset.features.get("label")
    label_names = getattr(label_feature, "names", None)
    dataset = dataset.cast_column("image", Image(decode=False))

    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(dataset):
        domain_name = normalize_domain_name(str(sample["domain"]))

        label_value = sample["label"]
        if isinstance(label_value, int) and label_names is not None:
            class_name = label_names[label_value]
        else:
            class_name = str(label_value)

        class_name = normalize_class_name(class_name)
        out_dir = target_dir / domain_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        image = sample["image"]
        image_bytes = image.get("bytes")
        if not image_bytes:
            raise ValueError(f"Office-Home row {idx} has no embedded image bytes")

        original_name = Path(image.get("path") or "").name
        if not original_name:
            original_name = f"{idx:06d}.jpg"
        out_path = out_dir / original_name
        if out_path.exists():
            raise FileExistsError(f"Duplicate Office-Home image path: {out_path}")
        out_path.write_bytes(image_bytes)

    print(f"Prepared Office-Home from HF dataset into {target_dir}")


if __name__ == "__main__":
    main()
