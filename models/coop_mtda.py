"""CoOp under source-only or joint mixed-target entropy training.

The prompt learner and frozen CLIP encoders are the original CoOp model. The
MTDA variant adds only conditional entropy on one concatenated target loader;
it does not construct pseudo labels or expose target-domain identities.
"""

import os.path as osp

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_pretrained_weights

from models.checkpoint_utils import load_checkpoint_compat
from models.coop import CustomCLIP, load_clip_to_cpu
from models.mtda_base import MultiTargetTrainerXU


def conditional_entropy(logits, eps):
    probabilities = F.softmax(logits.float(), dim=-1)
    return -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1).mean()


@TRAINER_REGISTRY.register()
class CoOpMTDA(MultiTargetTrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
        assert float(cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT) >= 0.0

    def build_model(self):
        cfg = self.cfg
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
            clip_model.float()
        self.model = CustomCLIP(cfg, self.dm.dataset.classnames, clip_model)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_("prompt_learner" in name)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.lambda_ent = float(cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT)
        self.entropy_eps = float(cfg.TRAINER.PROMPT_BASELINE_MTDA.ENTROPY_EPS)
        self.uses_target_training = self.lambda_ent > 0.0
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"CoOp mixed-target entropy weight: {self.lambda_ent}")
        print(f"Trainable parameter count: {trainable:,}")

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u = self.parse_batch_train(batch_x, batch_u)
        precision = self.cfg.TRAINER.COOP.PREC

        def compute_loss():
            logits_x = self.model(image_x)
            source_ce = F.cross_entropy(logits_x, label_x)
            target_entropy = source_ce.new_zeros(())
            if self.uses_target_training:
                if list(image_u) != ["mixed_target"]:
                    raise RuntimeError(f"Expected one mixed target batch, got {list(image_u)}")
                target_entropy = conditional_entropy(
                    self.model(image_u["mixed_target"]), self.entropy_eps
                )
            return logits_x, source_ce, target_entropy, (
                source_ce + self.lambda_ent * target_entropy
            )

        if precision == "amp":
            with autocast():
                logits_x, source_ce, target_entropy, loss = compute_loss()
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits_x, source_ce, target_entropy, loss = compute_loss()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return {
            "loss": loss.item(),
            "source_ce": source_ce.item(),
            "target_entropy": target_entropy.item(),
            "acc_src": compute_accuracy(logits_x, label_x)[0].item(),
        }

    def model_inference(self, input_tensor, domain_name=None):
        del domain_name
        return self.model(input_tensor)

    def load_model(self, directory, epoch=None):
        if not directory:
            return
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        for name in self.get_model_names():
            path = osp.join(directory, name, model_file)
            checkpoint = load_checkpoint_compat(path)
            state_dict = checkpoint["state_dict"]
            state_dict.pop("token_prefix", None)
            state_dict.pop("token_suffix", None)
            print(f'Loading weights to {name} from "{path}" (epoch={checkpoint["epoch"]})')
            self._models[name].load_state_dict(state_dict, strict=False)
