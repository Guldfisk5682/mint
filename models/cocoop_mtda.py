import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_pretrained_weights

from models.checkpoint_utils import load_checkpoint_compat
from models.cocoop import PromptLearner, TextEncoder, load_clip_to_cpu
from models.mtda_base import MultiTargetTrainerXU


class CustomCLIPMTDA(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        baseline_cfg = cfg.TRAINER.PROMPT_BASELINE_MTDA
        self.lambda_ent = float(baseline_cfg.LAMBDA_ENT)
        self.entropy_eps = float(baseline_cfg.ENTROPY_EPS)
        self.instance_chunk_size = int(
            cfg.TRAINER.COCOOP_MTDA.INSTANCE_CHUNK_SIZE
        )
        if self.instance_chunk_size < 1:
            raise ValueError("INSTANCE_CHUNK_SIZE must be at least 1")
        self.gradient_microbatch_size = int(
            cfg.TRAINER.COCOOP_MTDA.GRADIENT_MICROBATCH_SIZE
        )
        if self.gradient_microbatch_size < 0:
            raise ValueError("GRADIENT_MICROBATCH_SIZE cannot be negative")
        self.debug_print_once = cfg.TRAINER.COCOOP_MTDA.DEBUG.PRINT_ONCE
        self._has_printed_debug = False

    def _encode_logits(self, image_features):
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner(image_features)
        logit_scale = self.logit_scale.exp()

        logits = []
        num_classes = prompts.shape[1]
        for start in range(0, prompts.shape[0], self.instance_chunk_size):
            end = min(start + self.instance_chunk_size, prompts.shape[0])
            chunk_size = end - start
            prompt_chunk = prompts[start:end].reshape(
                chunk_size * num_classes, prompts.shape[2], prompts.shape[3]
            )
            token_chunk = self.tokenized_prompts.unsqueeze(0).expand(
                chunk_size, -1, -1
            ).reshape(chunk_size * num_classes, -1)
            text_features = self.text_encoder(prompt_chunk, token_chunk).reshape(
                chunk_size, num_classes, -1
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits.append(
                logit_scale
                * torch.einsum(
                    "bd,bcd->bc",
                    image_features_norm[start:end],
                    text_features,
                )
            )

        return torch.cat(logits, dim=0)

    def _build_debug_snapshot(self, image_s, image_u_dict, image_features, logits, loss):
        if not self.debug_print_once or self._has_printed_debug:
            return

        target_shapes = OrderedDict()
        for domain_name, image_u in image_u_dict.items():
            target_shapes[domain_name] = tuple(image_u.shape)

        print("[CoCoOpMTDA debug]")
        print("source domain:", self.cfg.DATASET.SOURCE_DOMAINS[0])
        print("target domains:", list(image_u_dict.keys()))
        print("source batch shape:", tuple(image_s.shape))
        for domain_name, shape in target_shapes.items():
            print(f"target batch shape [{domain_name}]:", shape)
        print("logits shape:", tuple(logits.shape))
        print("loss:", float(loss.detach().item()))
        self._has_printed_debug = True

    def forward_train(self, image_s, label_s, image_u_dict):
        image_features = self.image_encoder(image_s.type(self.dtype))
        logits = self._encode_logits(image_features)
        loss_ce = F.cross_entropy(logits, label_s)
        loss_ent = loss_ce.new_zeros(())
        if self.lambda_ent > 0.0:
            if list(image_u_dict) != ["mixed_target"]:
                raise RuntimeError(
                    f"Expected one mixed target batch, got {list(image_u_dict)}"
                )
            target_features = self.image_encoder(
                image_u_dict["mixed_target"].type(self.dtype)
            )
            target_logits = self._encode_logits(target_features)
            probabilities = F.softmax(target_logits.float(), dim=-1)
            loss_ent = -(
                probabilities * probabilities.clamp_min(self.entropy_eps).log()
            ).sum(dim=-1).mean()
        loss = loss_ce + self.lambda_ent * loss_ent
        self._build_debug_snapshot(image_s, image_u_dict, image_features, logits, loss)

        acc = compute_accuracy(logits, label_s)[0].item()
        return {
            "loss": loss,
            "loss_ce": loss_ce.detach(),
            "target_entropy": loss_ent.detach(),
            "acc_src": torch.tensor(acc, device=loss_ce.device),
        }

    def forward_inference(self, image, domain_name=None):
        image_features = self.image_encoder(image.type(self.dtype))
        return self._encode_logits(image_features)

    def forward(self, image, domain_name=None):
        return self.forward_inference(image, domain_name=domain_name)


@TRAINER_REGISTRY.register()
class CoCoOpMTDA(MultiTargetTrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP_MTDA.PREC in ["fp16", "fp32", "amp"]
        assert float(cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT) >= 0.0

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COCOOP_MTDA.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building CoCoOpMTDA")
        self.model = CustomCLIPMTDA(cfg, classnames, clip_model)
        self.gradient_microbatch_size = int(
            cfg.TRAINER.COCOOP_MTDA.GRADIENT_MICROBATCH_SIZE
        )
        self.uses_target_training = (
            float(cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT) > 0.0
        )

        print("Turning off gradients in the CLIP image/text encoders")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        enabled = sorted(
            name for name, param in self.model.named_parameters() if param.requires_grad
        )
        print("Parameters to be updated:")
        for name in enabled:
            print(f"  - {name}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        print(
            "CoCoOp mixed-target entropy weight: "
            f"{cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT}"
        )
        print(
            "CoCoOp instance prompt encoder chunk size: "
            f"{cfg.TRAINER.COCOOP_MTDA.INSTANCE_CHUNK_SIZE}"
        )
        print(
            "CoCoOp gradient microbatch size: "
            f"{cfg.TRAINER.COCOOP_MTDA.GRADIENT_MICROBATCH_SIZE}"
        )
        print(f"Trainable parameter count: {trainable_params:,}")
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("cocoop_mtda", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP_MTDA.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def _model_ref(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u = self.parse_batch_train(batch_x, batch_u)
        model = self._model_ref()
        prec = self.cfg.TRAINER.COCOOP_MTDA.PREC
        logical_batch_size = image_x.shape[0]
        microbatch_size = self.gradient_microbatch_size or logical_batch_size
        microbatch_size = min(microbatch_size, logical_batch_size)
        loss_summary = {}

        self.optim.zero_grad()
        for start in range(0, logical_batch_size, microbatch_size):
            end = min(start + microbatch_size, logical_batch_size)
            weight = (end - start) / logical_batch_size
            image_u_micro = {
                name: images[start:end] for name, images in image_u.items()
            }

            if prec == "amp":
                with autocast():
                    outputs = model.forward_train(
                        image_x[start:end],
                        label_x[start:end],
                        image_u_micro,
                    )
                    weighted_loss = outputs["loss"] * weight
                self.scaler.scale(weighted_loss).backward()
            else:
                outputs = model.forward_train(
                    image_x[start:end],
                    label_x[start:end],
                    image_u_micro,
                )
                weighted_loss = outputs["loss"] * weight
                weighted_loss.backward()

            for key, value in outputs.items():
                scalar = value.item() if torch.is_tensor(value) else float(value)
                loss_summary[key] = loss_summary.get(key, 0.0) + weight * scalar

        if prec == "amp":
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input_tensor, domain_name=None):
        return self._model_ref().forward_inference(input_tensor, domain_name=domain_name)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint_compat(model_path)
            state_dict = checkpoint["state_dict"]
            state_dict.pop("prompt_learner.token_prefix", None)
            state_dict.pop("prompt_learner.token_suffix", None)

            loaded_epoch = checkpoint["epoch"]
            print(f'Loading weights to {name} from "{model_path}" (epoch = {loaded_epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
