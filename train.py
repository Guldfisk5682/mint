import argparse
import datetime
import inspect
import json
import os
import socket
import subprocess
import sys
import torch


def patch_torch_lr_scheduler_compatibility():
    """Keep the legacy Dassl warmup scheduler runnable on PyTorch >= 2.9.

    Dassl's ``ConstantWarmupScheduler`` still forwards ``verbose`` to
    ``LRScheduler.__init__``. PyTorch 2.9 removed that argument while keeping
    the rest of the scheduler API unchanged. Patch only that removed argument
    and leave pre-2.9 installations untouched.
    """

    scheduler_cls = torch.optim.lr_scheduler.LRScheduler
    parameters = inspect.signature(scheduler_cls.__init__).parameters
    if "verbose" in parameters:
        return

    original_init = scheduler_cls.__init__
    if getattr(original_init, "_dassl_legacy_verbose_compat", False):
        return

    def compatible_init(self, optimizer, last_epoch=-1, verbose=None):
        return original_init(self, optimizer, last_epoch)

    compatible_init._dassl_legacy_verbose_compat = True
    scheduler_cls.__init__ = compatible_init
    print("Applied PyTorch >=2.9 Dassl LRScheduler compatibility patch")


patch_torch_lr_scheduler_compatibility()

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# SS-MTDA dataset and trainer registry entries.
import datasets.office31
import datasets.office_home_mtda
import datasets.domainnet_mtda

import models.cocoop
import models.esmaple
import models.mint
import models.maple


def _git_output(*args):
    try:
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "<unavailable>"


def configure_reproducibility(seed):
    if seed < 0:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def append_run_metadata(cfg):
    metadata = {
        "argv": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "cwd": os.getcwd(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_short": _git_output("status", "--short"),
        "hostname": socket.gethostname(),
        "output_dir": cfg.OUTPUT_DIR,
        "python": sys.version.split()[0],
        "seed": cfg.SEED,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch": str(torch.__version__),
        "trainer": cfg.TRAINER.NAME,
    }
    path = os.path.join(cfg.OUTPUT_DIR, "run_metadata.jsonl")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
    print(f"Appended run metadata: {path}")


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    # A positive value caps realized optimizer steps per epoch while preserving
    # the trainer's existing epoch-based scheduler interface.
    cfg.TRAIN.MAX_BATCHES_PER_EPOCH = -1
    cfg.TRAIN.SOURCE_ONLY = False

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.COCOOP_MTDA = CN()
    cfg.TRAINER.COCOOP_MTDA.PREC = "fp16"
    # Number of image-conditioned class-prompt groups encoded together.
    # A value of 1 preserves the original per-image execution order.
    cfg.TRAINER.COCOOP_MTDA.INSTANCE_CHUNK_SIZE = 1
    # Split a logical loader batch for activation memory, then accumulate all
    # gradients before one optimizer step. Zero disables microbatching.
    cfg.TRAINER.COCOOP_MTDA.GRADIENT_MICROBATCH_SIZE = 0
    cfg.TRAINER.COCOOP_MTDA.DEBUG = CN()
    cfg.TRAINER.COCOOP_MTDA.DEBUG.PRINT_ONCE = False

    cfg.TRAINER.PROMPT_BASELINE_MTDA = CN()
    cfg.TRAINER.PROMPT_BASELINE_MTDA.MIX_TARGETS = False
    cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT = 0.0
    cfg.TRAINER.PROMPT_BASELINE_MTDA.ENTROPY_EPS = 1e-8

    cfg.TRAINER.MAPLE_MTDA = CN()
    cfg.TRAINER.MAPLE_MTDA.N_CTX = 2
    cfg.TRAINER.MAPLE_MTDA.CTX_INIT = "a photo of a"
    cfg.TRAINER.MAPLE_MTDA.PREC = "fp16"
    cfg.TRAINER.MAPLE_MTDA.PROMPT_DEPTH = 9
    cfg.TRAINER.MAPLE_MTDA.USE_MAPLE_CLIP_BUILD = True
    cfg.TRAINER.MAPLE_MTDA.LAMBDA_PL = 0.0
    cfg.TRAINER.MAPLE_MTDA.LAMBDA_PL_FINAL = 0.0
    cfg.TRAINER.MAPLE_MTDA.PL_SCHEDULE = "constant"
    cfg.TRAINER.MAPLE_MTDA.PL_THRESHOLD = 0.7
    cfg.TRAINER.MAPLE_MTDA.PL_STUDENT_THRESHOLD = 0.7
    cfg.TRAINER.MAPLE_MTDA.PL_USE_STUDENT_LOW_CONF_MASK = True
    cfg.TRAINER.MAPLE_MTDA.PL_VARIANT = "legacy"
    cfg.TRAINER.MAPLE_MTDA.PL_DUAL_CONF_THRESHOLD = 0.7
    cfg.TRAINER.MAPLE_MTDA.PL_SOFT_BETA = 1.0
    cfg.TRAINER.MAPLE_MTDA.PL_STUDENT_SOFT_LAMBDA = 0.5
    cfg.TRAINER.MAPLE_MTDA.PL_STRONG_AUGMENT = "randaugment_fixmatch"
    cfg.TRAINER.MAPLE_MTDA.PL_DIAGNOSTIC_WINDOW = 50
    cfg.TRAINER.MAPLE_MTDA.PL_GRAD_AUDIT_INTERVAL = 50
    cfg.TRAINER.MAPLE_MTDA.ZS_PROMPT_TEMPLATE = "a photo of a {}."
    cfg.TRAINER.MAPLE_MTDA.WEAK_PL = CN()
    cfg.TRAINER.MAPLE_MTDA.WEAK_PL.ENABLED = False
    cfg.TRAINER.MAPLE_MTDA.WEAK_PL.LAMBDA = 0.0
    cfg.TRAINER.MAPLE_MTDA.WEAK_PL.TEACHER_THRESHOLD = 0.6
    cfg.TRAINER.MAPLE_MTDA.WEAK_PL.TEACHER_THRESHOLD_HIGH = 0.7
    cfg.TRAINER.MAPLE_MTDA.WEAK_PL.STUDENT_THRESHOLD = 0.7
    cfg.TRAINER.MAPLE_MTDA.WEAK_PL.USE_STUDENT_LOW_CONF_MASK = True
    cfg.TRAINER.MAPLE_MTDA.WEAK_PL.FRACTION = 0.2
    cfg.TRAINER.MAPLE_MTDA.WEAK_PL.MOMENTUM = 0.9
    cfg.TRAINER.MAPLE_MTDA.WEAK_PL.EPS = 1e-6
    cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL = CN()
    cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL.ENABLED = False
    cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL.LAMBDA = 0.0
    cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL.MODE = "confidence_band"
    cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL.TEMPERATURE = 2.0
    cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL.OLD_CONF_LOW = 0.45
    cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL.OLD_CONF_HIGH = 0.8
    cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL.CLIP_CONF_HIGH = 0.7
    cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL.EPS = 1e-6
    cfg.TRAINER.MAPLE_MTDA.POST_INIT = CN()
    cfg.TRAINER.MAPLE_MTDA.POST_INIT.ENABLED = False
    cfg.TRAINER.MAPLE_MTDA.POST_INIT.MODEL_DIR = ""
    cfg.TRAINER.MAPLE_MTDA.POST_INIT.CHECKPOINT_MODEL_NAME = ""
    cfg.TRAINER.MAPLE_MTDA.POST_INIT.LOAD_EPOCH = -1
    cfg.TRAINER.MAPLE_MTDA.GAP_CTX = CN()
    cfg.TRAINER.MAPLE_MTDA.GAP_CTX.STYLE_LAYERS = [3, 6, 9, 12]
    cfg.TRAINER.MAPLE_MTDA.GAP_CTX.HIDDEN_DIM = 512
    cfg.TRAINER.MAPLE_MTDA.GAP_CTX.ALPHA = 0.1
    cfg.TRAINER.MAPLE_MTDA.GAP_CTX.STYLE_EPS = 1e-6
    cfg.TRAINER.MAPLE_MTDA.DEBUG = CN()
    cfg.TRAINER.MAPLE_MTDA.DEBUG.PRINT_ONCE = False

    cfg.TRAINER.MAPLE_MTDA.CURRICULUM = CN()
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.DOMAIN_ORDER = []
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.STAGE_LIMIT = -1
    # Optional positive relative weights for unequal curriculum stage budgets.
    # Empty preserves the historical equal split.
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.STAGE_STEP_WEIGHTS = []
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.MICROBATCHES_PER_STEP = 3
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.RESET_OPTIM_PER_STAGE = False
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.STAGE_VIRTUAL_EPOCHS = 5
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY = CN()
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.ENABLED = False
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TOPK_PER_CLASS = 8
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.STUDENT_THRESHOLD = 0.7
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.CLIP_THRESHOLD = 0.7
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.LAMBDA = 1.0
    # Diagnostic controls are deliberately inert by default. ``online`` +
    # ``pseudo`` + ``one_pass`` exactly preserves the original replay path.
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.SELECTION_MODE = "online"
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.LABEL_SOURCE = "pseudo"
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.MANIFEST_PATH = ""
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TRAVERSAL = "one_pass"
    # none | one_pass_steps. The latter is a cycle-only diagnostic that
    # preserves one-pass's nominal number of replay-weighted optimizer steps.
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.NORMALIZATION = "none"
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.DIAGNOSTICS = CN()
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.DIAGNOSTICS.ENABLED = False
    cfg.TRAINER.MAPLE_MTDA.CURRICULUM.DIAGNOSTICS.AUDIT_ALL_DOMAINS = True

    # Diagnostic-only switch. When enabled, MTDA trainers evaluate target
    # domains after every epoch to expose accuracy curves; normal experiments
    # should keep the default final-test-only behavior.
    cfg.TEST.EVAL_EVERY_EPOCH = False

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    configure_reproducibility(cfg.SEED)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    append_run_metadata(cfg)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        # Respect CUDA_VISIBLE_DEVICES by binding the current process
        # to the first visible logical GPU before any model/data placement.
        torch.cuda.set_device(0)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            "Using CUDA device: logical cuda:0"
            " "
            f"(CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', '<unset>')}, "
            f"CUDA_VISIBLE_DEVICES={visible}, "
            f"torch.current_device()={torch.cuda.current_device()}, "
            f"torch.cuda.device_count()={torch.cuda.device_count()})"
        )
        if cfg.SEED >= 0:
            print("Deterministic CUDA execution enabled for fixed-seed runs")

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    main(args)
