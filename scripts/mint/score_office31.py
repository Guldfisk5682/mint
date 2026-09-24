#!/usr/bin/env python3
"""Rank the two Office-31 targets by source-only prediction entropy."""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms import build_transform
from dassl.engine import build_trainer
from dassl.utils import set_random_seed

import train


DOMAINS = {"A": "amazon", "D": "dslr", "W": "webcam"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=list(DOMAINS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--load-epoch", type=int, default=5)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def build_cfg(args):
    targets = [domain for code, domain in DOMAINS.items() if code != args.source]
    cfg_args = SimpleNamespace(
        root=args.root,
        output_dir=str(Path(args.output).parent / "difficulty_tmp"),
        resume="",
        seed=args.seed,
        source_domains=[DOMAINS[args.source]],
        target_domains=targets,
        transforms=None,
        trainer="ContinuousSharedProjMaPLeMTDA",
        backbone="",
        head="",
        dataset_config_file="config/datasets/office31_mtda.yaml",
        config_file=(
            "config/trainers/esmaple.yaml"
        ),
        opts=[
            "DATALOADER.TEST.BATCH_SIZE",
            str(args.batch_size),
            "DATALOADER.NUM_WORKERS",
            str(args.num_workers),
        ],
    )
    return train.setup_cfg(cfg_args)


@torch.no_grad()
def score_domain(trainer, loader):
    entropy_sum = 0.0
    confidence_sum = 0.0
    sample_count = 0
    trainer.set_model_mode("eval")
    for batch in loader:
        image = batch["img"].to(trainer.device)
        probabilities = trainer.model(image).float().softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        entropy_sum += float(entropy.sum().item())
        confidence_sum += float(probabilities.max(dim=-1).values.sum().item())
        sample_count += int(image.shape[0])
    return {
        "samples": sample_count,
        "mean_entropy": entropy_sum / sample_count,
        "mean_normalized_entropy": entropy_sum / (sample_count * math.log(31)),
        "mean_confidence": confidence_sum / sample_count,
    }


def build_target_train_score_loaders(trainer):
    """Score only the unlabeled target-training split with evaluation transforms."""
    transform = build_transform(trainer.cfg, is_train=False)
    return {
        domain: build_data_loader(
            trainer.cfg,
            sampler_type="SequentialSampler",
            data_source=items,
            batch_size=trainer.cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=transform,
            is_train=False,
        )
        for domain, items in trainer.dm.dataset.train_u_by_domain.items()
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)
    trainer = build_trainer(build_cfg(args))
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    scores = {
        domain: score_domain(trainer, loader)
        for domain, loader in build_target_train_score_loaders(trainer).items()
    }
    easy_to_hard = sorted(
        scores,
        key=lambda domain: (scores[domain]["mean_normalized_entropy"], domain),
    )
    payload = {
        "source": DOMAINS[args.source],
        "seed": args.seed,
        "scoring_split": "target_train",
        "selection_uses_target_labels": False,
        "selection_uses_target_test_inputs": False,
        "scores": scores,
        "easy_to_hard": easy_to_hard,
        "hard_to_easy": list(reversed(easy_to_hard)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
