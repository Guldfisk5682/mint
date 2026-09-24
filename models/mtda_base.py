import datetime
import time
from collections import OrderedDict

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from dassl.data.data_manager import DatasetWrapper, build_data_loader
from dassl.data.datasets import build_dataset
from dassl.data.transforms import build_transform
from dassl.engine.trainer import SimpleTrainer
from dassl.utils import AverageMeter, MetricMeter


def cap_num_batches(cfg, natural_batches, context):
    """Apply an optional fixed per-epoch optimizer-step ceiling."""
    cap = int(getattr(cfg.TRAIN, "MAX_BATCHES_PER_EPOCH", -1))
    if cap == 0 or cap < -1:
        raise ValueError("TRAIN.MAX_BATCHES_PER_EPOCH must be -1 or positive")
    realized = min(natural_batches, cap) if cap > 0 else natural_batches
    if realized <= 0:
        raise RuntimeError(f"{context} resolved to zero optimizer steps")
    if cap > 0:
        print(
            f"Fixed-step budget [{context}]: natural_batches={natural_batches}, "
            f"cap={cap}, realized_batches={realized}"
        )
    return realized


class MultiTargetDataManager:
    """Build one unlabeled loader and one test loader per target domain."""

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_train_x=None,
        custom_tfm_train_u=None,
        custom_tfm_test=None,
        dataset_wrapper=None,
    ):
        dataset = build_dataset(cfg)

        if not hasattr(dataset, "train_u_by_domain") or not hasattr(dataset, "test_by_domain"):
            raise TypeError(
                "MultiTargetDataManager expects a dataset exposing "
                "train_u_by_domain and test_by_domain"
            )

        if custom_tfm_train is not None and (
            custom_tfm_train_x is not None or custom_tfm_train_u is not None
        ):
            raise ValueError(
                "custom_tfm_train cannot be combined with split x/u transforms"
            )
        if custom_tfm_train is not None:
            print("* Using custom transform for training")
            tfm_train_x = custom_tfm_train
            tfm_train_u = custom_tfm_train
        else:
            tfm_train_x = (
                build_transform(cfg, is_train=True)
                if custom_tfm_train_x is None
                else custom_tfm_train_x
            )
            tfm_train_u = (
                tfm_train_x
                if custom_tfm_train_u is None
                else custom_tfm_train_u
            )
            if custom_tfm_train_x is not None or custom_tfm_train_u is not None:
                print("* Using separate source/target transforms for training")

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        if dataset_wrapper is None:
            dataset_wrapper = DatasetWrapper

        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train_x,
            is_train=True,
            dataset_wrapper=dataset_wrapper,
        )

        sampler_type_u = cfg.DATALOADER.TRAIN_U.SAMPLER
        batch_size_u = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
        n_domain_u = cfg.DATALOADER.TRAIN_U.N_DOMAIN
        n_ins_u = cfg.DATALOADER.TRAIN_U.N_INS
        if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
            sampler_type_u = cfg.DATALOADER.TRAIN_X.SAMPLER
            batch_size_u = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
            n_domain_u = cfg.DATALOADER.TRAIN_X.N_DOMAIN
            n_ins_u = cfg.DATALOADER.TRAIN_X.N_INS

        target_training_sets = dataset.train_u_by_domain
        mix_targets = bool(
            getattr(
                getattr(cfg.TRAINER, "PROMPT_BASELINE_MTDA", None),
                "MIX_TARGETS",
                False,
            )
        )
        if mix_targets:
            entropy_weight = float(
                cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT
            )
            if entropy_weight == 0.0:
                target_training_sets = OrderedDict()
                print("Source-only baseline: target training loader is not built")
            else:
                mixed_items = []
                for items in target_training_sets.values():
                    mixed_items.extend(items)
                target_training_sets = OrderedDict([("mixed_target", mixed_items)])
                print(
                    "Mixed-target training loader: {} images from {} domains".format(
                        len(mixed_items), len(dataset.train_u_by_domain)
                    )
                )

        train_loaders_u = OrderedDict()
        for domain_name, items in target_training_sets.items():
            train_loaders_u[domain_name] = build_data_loader(
                cfg,
                sampler_type=sampler_type_u,
                data_source=items,
                batch_size=batch_size_u,
                n_domain=n_domain_u,
                n_ins=n_ins_u,
                tfm=tfm_train_u,
                is_train=True,
                dataset_wrapper=dataset_wrapper,
            )

        test_loaders_by_domain = OrderedDict()
        for domain_name, items in dataset.test_by_domain.items():
            test_loaders_by_domain[domain_name] = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=items,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loaders_u
        self.train_loaders_u = train_loaders_u
        self.val_loader = None
        self.test_loader = next(iter(test_loaders_by_domain.values()))
        self.test_loaders_by_domain = test_loaders_by_domain

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        table = [
            ["Dataset", cfg.DATASET.NAME],
            ["Source", cfg.DATASET.SOURCE_DOMAINS],
            ["Targets", list(self.train_loaders_u.keys())],
            ["# classes", f"{self.num_classes:,}"],
            ["# train_x", f"{len(self.dataset.train_x):,}"],
        ]
        for domain_name, items in self.dataset.train_u_by_domain.items():
            table.append([f"# train_u[{domain_name}]", f"{len(items):,}"])
        for domain_name, items in self.dataset.test_by_domain.items():
            table.append([f"# test[{domain_name}]", f"{len(items):,}"])
        print(tabulate(table))


class MultiTargetTrainerXU(SimpleTrainer):
    """A simple trainer base for one labeled source and multiple unlabeled targets."""

    def build_data_loader(self):
        dm = MultiTargetDataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.test_loaders_by_domain = dm.test_loaders_by_domain

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm

    def after_epoch(self):
        if self.cfg.TEST.FINAL_MODEL != "last_step":
            raise ValueError(
                "SS-MTDA runs require TEST.FINAL_MODEL=last_step; "
                "target-domain accuracy must not select checkpoints"
            )
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        eval_every_epoch = bool(getattr(self.cfg.TEST, "EVAL_EVERY_EPOCH", False))
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and eval_every_epoch:
            self.test()
            self._last_eval_epoch = self.epoch

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            print("Deploy the last-epoch model")

            already_evaluated_last_epoch = (
                bool(getattr(self.cfg.TEST, "EVAL_EVERY_EPOCH", False))
                and getattr(self, "_last_eval_epoch", None) == self.epoch
            )
            if already_evaluated_last_epoch:
                print("Skip final test because the last epoch was already evaluated")
            else:
                self.test()

        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        self.close_writer()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        len_train_loader_x = len(self.train_loader_x)
        use_target_training = not bool(
            getattr(self.cfg.TRAIN, "SOURCE_ONLY", False)
        ) and bool(getattr(self, "uses_target_training", True))

        if not use_target_training:
            self.num_batches = len_train_loader_x
        else:
            len_train_loader_u = [
                len(loader) for loader in self.train_loader_u.values()
            ]
            if not len_train_loader_u:
                raise RuntimeError("Target training enabled but no target loader was built")
            min_train_loader_u = min(len_train_loader_u)
            if self.cfg.TRAIN.COUNT_ITER == "train_x":
                self.num_batches = len_train_loader_x
            elif self.cfg.TRAIN.COUNT_ITER == "train_u":
                self.num_batches = min_train_loader_u
            elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
                self.num_batches = min([len_train_loader_x, *len_train_loader_u])
            else:
                raise ValueError(
                    f"Unsupported TRAIN.COUNT_ITER={self.cfg.TRAIN.COUNT_ITER}"
                )

        self.num_batches = cap_num_batches(
            self.cfg, self.num_batches, "MultiTargetTrainerXU"
        )

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iters = OrderedDict()
        if use_target_training:
            train_loader_u_iters = OrderedDict(
                (domain_name, iter(loader))
                for domain_name, loader in self.train_loader_u.items()
            )

        if self.epoch == 0:
            print(
                "Baseline budget audit: epochs={}, steps_per_epoch={}, "
                "optimizer_steps={}, target_training={}".format(
                    self.max_epoch,
                    self.num_batches,
                    self.max_epoch * self.num_batches,
                    use_target_training,
                )
            )

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            batch_u = OrderedDict()
            for domain_name, loader in (
                self.train_loader_u.items() if use_target_training else []
            ):
                try:
                    batch_u[domain_name] = next(train_loader_u_iters[domain_name])
                except StopIteration:
                    train_loader_u_iters[domain_name] = iter(loader)
                    batch_u[domain_name] = next(train_loader_u_iters[domain_name])

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"{losses}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}",
                ]
                print(" ".join(info))

            # Result directories can live on network-backed storage. Writing
            # TensorBoard events every optimizer step adds synchronous I/O
            # without increasing the resolution of the printed training log.
            # Keep the same metrics, sampled at PRINT_FREQ and the epoch end.
            if meet_freq or (self.batch_idx + 1) == self.num_batches:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                for name, meter in losses.meters.items():
                    self.write_scalar("train/" + name, meter.avg, n_iter)
                self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"].to(self.device)
        label_x = batch_x["label"].to(self.device)

        input_u = OrderedDict()
        for domain_name, batch in batch_u.items():
            input_u[domain_name] = batch["img"].to(self.device)

        return input_x, label_x, input_u

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")

        target_results = OrderedDict()
        for domain_name, data_loader in self.test_loaders_by_domain.items():
            print(f"Evaluate on target domain: {domain_name}")
            self.evaluator.reset()
            for batch in tqdm(data_loader):
                input_tensor, label = self.parse_batch_test(batch)
                output = self.model_inference(input_tensor, domain_name=domain_name)
                self.evaluator.process(output, label)
            results = self.evaluator.evaluate()
            acc = results["accuracy"]
            target_results[domain_name] = acc
            print(f"Target domain {domain_name} accuracy: {acc:.2f}%")
            for key, value in results.items():
                self.write_scalar(f"test/{domain_name}_{key}", value, self.epoch)

        macro_avg = float(np.mean(list(target_results.values()))) if target_results else 0.0
        print(f"Per-source macro average: {macro_avg:.2f}%")
        print(f"Overall average: {macro_avg:.2f}%")
        self.write_scalar("test/macro_avg", macro_avg, self.epoch)
        return macro_avg

    def model_inference(self, input_tensor, domain_name=None):
        if domain_name is None:
            return self.model(input_tensor)
        return self.model(input_tensor, domain_name=domain_name)
