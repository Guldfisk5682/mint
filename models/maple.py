import copy
import math
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.engine import TRAINER_REGISTRY
from dassl.data.transforms import build_transform
from dassl.optim import build_lr_scheduler, build_optimizer

from models.checkpoint_utils import load_checkpoint_compat, load_state_dict_checked
from models.cocoop import load_clip_to_cpu as load_base_clip_to_cpu
from models.mtda_base import MultiTargetDataManager, MultiTargetTrainerXU

_tokenizer = _Tokenizer()


def build_self_distill_mask(
    mode,
    old_conf,
    *,
    old_conf_low,
    old_conf_high,
    reference_conf=None,
    clip_conf_high=0.7,
):
    if mode == "all":
        return torch.ones_like(old_conf, dtype=torch.bool)
    if mode == "confidence_band":
        return old_conf.ge(old_conf_low) & old_conf.lt(old_conf_high)
    if mode == "teacher_handoff":
        if reference_conf is None:
            raise RuntimeError(
                "teacher_handoff self-distillation requires frozen-CLIP confidence"
            )
        return old_conf.ge(old_conf_low) & reference_conf.lt(clip_conf_high)
    raise ValueError(f"Unsupported MAPLE_MTDA.SELF_DISTILL.MODE={mode}")


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def load_clip_to_cpu(cfg):
    design_details = {
        "trainer": "MaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.MAPLE_MTDA.N_CTX,
    }

    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoderMaPLe(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer([x, compound_prompts_deeper_text, 0])[0]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection
        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        maple_cfg = cfg.TRAINER.MAPLE_MTDA
        n_cls = len(classnames)
        n_ctx = maple_cfg.N_CTX
        ctx_init = maple_cfg.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.conv1.out_channels
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        if cfg_imsize != clip_imsize:
            raise ValueError(f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})")
        if maple_cfg.PROMPT_DEPTH < 1:
            raise ValueError("TRAINER.MAPLE_MTDA.PROMPT_DEPTH should be >= 1")

        self.compound_prompts_depth = maple_cfg.PROMPT_DEPTH

        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        self.proj = nn.Linear(ctx_dim, visual_dim).to(dtype=dtype)

        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for prompt in self.compound_prompts_text:
            nn.init.normal_(prompt, std=0.02)

        single_layer = nn.Linear(ctx_dim, visual_dim).to(dtype=dtype)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

        print("MaPLeMTDA design: source-only multi-modal prompt baseline")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        print(f"Prompt depth: {self.compound_prompts_depth}")

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        visual_deep_prompts = [
            layer(self.compound_prompts_text[index])
            for index, layer in enumerate(self.compound_prompt_projections)
        ]
        return (
            prompts,
            self.proj(self.ctx),
            self.compound_prompts_text,
            visual_deep_prompts,
        )


class CustomMaPLeMTDA(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        maple_cfg = cfg.TRAINER.MAPLE_MTDA
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoderMaPLe(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        baseline_cfg = cfg.TRAINER.PROMPT_BASELINE_MTDA
        self.lambda_ent = float(baseline_cfg.LAMBDA_ENT)
        self.entropy_eps = float(baseline_cfg.ENTROPY_EPS)
        self.lambda_pl = float(maple_cfg.LAMBDA_PL)
        self.lambda_pl_final = float(maple_cfg.LAMBDA_PL_FINAL)
        self.pl_schedule = str(maple_cfg.PL_SCHEDULE).lower()
        self._pl_progress = 0.0
        self.pl_threshold = float(maple_cfg.PL_THRESHOLD)
        self.pl_student_threshold = float(maple_cfg.PL_STUDENT_THRESHOLD)
        self.pl_use_student_low_conf_mask = bool(
            maple_cfg.PL_USE_STUDENT_LOW_CONF_MASK
        )
        self._init_dual_pl_config(maple_cfg)
        self._init_weak_pl_config(maple_cfg)
        self._init_self_distill_config(maple_cfg)
        self.debug_print_once = bool(maple_cfg.DEBUG.PRINT_ONCE)
        self._debug_printed = False

        zs_template = str(maple_cfg.ZS_PROMPT_TEMPLATE)
        if "{}" not in zs_template:
            raise ValueError("TRAINER.MAPLE_MTDA.ZS_PROMPT_TEMPLATE must contain '{}'")
        classnames = [name.replace("_", " ") for name in classnames]
        zs_prompts = [zs_template.format(name) for name in classnames]
        self.register_buffer(
            "zs_tokenized_prompts",
            torch.cat([clip.tokenize(prompt) for prompt in zs_prompts]),
        )

        log_prefix = getattr(self, "log_prefix", "MaPLeMTDA")
        print(f"{log_prefix} pseudo-label weight: {self.lambda_pl}")
        print(f"{log_prefix} pseudo-label final weight: {self.lambda_pl_final}")
        print(f"{log_prefix} pseudo-label schedule: {self.pl_schedule}")
        print(f"{log_prefix} pseudo-label threshold: {self.pl_threshold}")
        print(f"{log_prefix} pseudo-label student threshold: {self.pl_student_threshold}")
        print(
            f"{log_prefix} pseudo-label low-conf only: "
            f"{self.pl_use_student_low_conf_mask}"
        )
        print(f"{log_prefix} pseudo-label variant: {self.pl_variant}")
        print(
            f"{log_prefix} dual-confidence threshold: "
            f"{self.pl_dual_conf_threshold}"
        )
        print(f"{log_prefix} soft pseudo-label beta: {self.pl_soft_beta}")
        print(
            f"{log_prefix} student-soft weight: "
            f"{self.pl_student_soft_lambda}"
        )
        print(f"{log_prefix} weak PL enabled: {self.weak_pl_enabled}")
        print(f"{log_prefix} weak PL weight: {self.lambda_weak_pl}")
        print(
            f"{log_prefix} weak PL teacher thresholds: "
            f"[{self.weak_pl_teacher_threshold}, {self.weak_pl_teacher_threshold_high})"
        )
        print(f"{log_prefix} weak PL student threshold: {self.weak_pl_student_threshold}")
        print(f"{log_prefix} weak PL class fraction: {self.weak_pl_fraction}")
        print(f"{log_prefix} self-distill enabled: {self.self_distill_enabled}")
        print(f"{log_prefix} self-distill weight: {self.lambda_self_distill}")
        print(f"{log_prefix} self-distill mode: {self.self_distill_mode}")
        print(f"{log_prefix} self-distill temperature: {self.self_distill_temperature}")
        print(
            f"{log_prefix} self-distill old confidence band: "
            f"[{self.self_distill_old_conf_low}, {self.self_distill_old_conf_high})"
        )
        print(f"{log_prefix} zero-shot prompt template: {zs_template}")

    def set_training_progress(self, progress):
        self._pl_progress = max(0.0, min(1.0, float(progress)))

    def current_lambda_pl(self):
        if self.pl_schedule == "constant":
            return self.lambda_pl
        if self.pl_schedule == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * self._pl_progress))
            return self.lambda_pl_final + (self.lambda_pl - self.lambda_pl_final) * cosine
        raise ValueError(f"Unsupported MAPLE_MTDA.PL_SCHEDULE={self.pl_schedule}")

    @staticmethod
    def _ensure_finite(name, tensor):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values detected in {name}")

    def _init_dual_pl_config(self, maple_cfg):
        self.pl_variant = str(maple_cfg.PL_VARIANT).lower()
        self.pl_dual_conf_threshold = float(maple_cfg.PL_DUAL_CONF_THRESHOLD)
        self.pl_soft_beta = float(maple_cfg.PL_SOFT_BETA)
        self.pl_student_soft_lambda = float(maple_cfg.PL_STUDENT_SOFT_LAMBDA)

    def _init_weak_pl_config(self, maple_cfg):
        weak_cfg = maple_cfg.WEAK_PL
        self.weak_pl_enabled = bool(weak_cfg.ENABLED)
        self.lambda_weak_pl = float(weak_cfg.LAMBDA)
        self.weak_pl_teacher_threshold = float(weak_cfg.TEACHER_THRESHOLD)
        self.weak_pl_teacher_threshold_high = float(
            weak_cfg.TEACHER_THRESHOLD_HIGH
        )
        self.weak_pl_student_threshold = float(weak_cfg.STUDENT_THRESHOLD)
        self.weak_pl_use_student_low_conf_mask = bool(
            weak_cfg.USE_STUDENT_LOW_CONF_MASK
        )
        self.weak_pl_fraction = float(weak_cfg.FRACTION)
        self.weak_pl_momentum = float(weak_cfg.MOMENTUM)
        self.weak_pl_eps = float(weak_cfg.EPS)
        self._weak_pl_domain_state = {}

    def _init_self_distill_config(self, maple_cfg):
        sd_cfg = maple_cfg.SELF_DISTILL
        self.self_distill_enabled = bool(sd_cfg.ENABLED)
        self.lambda_self_distill = float(sd_cfg.LAMBDA)
        self.self_distill_mode = str(sd_cfg.MODE).lower()
        self.self_distill_temperature = float(sd_cfg.TEMPERATURE)
        self.self_distill_old_conf_low = float(sd_cfg.OLD_CONF_LOW)
        self.self_distill_old_conf_high = float(sd_cfg.OLD_CONF_HIGH)
        self.self_distill_clip_conf_high = float(sd_cfg.CLIP_CONF_HIGH)
        self.self_distill_eps = float(sd_cfg.EPS)
        self._self_distill_old_model_holder = [None]

    def build_self_distill_old_model(self):
        old_existing = self._self_distill_old_model_holder[0]
        self._self_distill_old_model_holder[0] = None
        old_model = copy.deepcopy(self)
        self._self_distill_old_model_holder[0] = old_existing
        old_model.self_distill_enabled = False
        old_model.lambda_self_distill = 0.0
        old_model._self_distill_old_model_holder[0] = None
        old_model.to(next(self.parameters()).device)
        old_model.eval()
        for param in old_model.parameters():
            param.requires_grad_(False)
        self._self_distill_old_model_holder[0] = old_model

    @staticmethod
    def _masked_mean(values, mask):
        mask_float = mask.float()
        denom = mask_float.sum()
        if denom.item() <= 0:
            return values.new_zeros(())
        return (values * mask_float).sum() / denom.clamp_min(1.0)

    def forward(self, image, domain_name=None):
        del domain_name
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        (
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = self.prompt_learner()

        text_features = self.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text
        )
        image_features = self.image_encoder(
            image.type(self.dtype), shared_ctx, deep_compound_prompts_vision
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        return logits

    @torch.no_grad()
    def _encode_reference_image(self, image):
        visual = self.image_encoder
        x = visual.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer([x, [], 0])[0]
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj
        x = x.float()
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._ensure_finite("maple_reference_image_features", x)
        return x

    @torch.no_grad()
    def _zero_shot_text_features(self, device):
        tokenized_prompts = self.zs_tokenized_prompts.to(device)
        prompts = self.token_embedding(tokenized_prompts).type(self.dtype)
        text_features = self.text_encoder(prompts, tokenized_prompts, []).float()
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        self._ensure_finite("maple_zero_shot_text_features", text_features)
        return text_features

    @torch.no_grad()
    def _compute_reference_logits(self, image):
        image_features = self._encode_reference_image(image)
        text_features = self._zero_shot_text_features(image_features.device)
        logits = self.logit_scale.float().exp() * image_features @ text_features.t()
        self._ensure_finite("maple_reference_logits", logits)
        return logits.detach()

    def _pseudo_label_loss(self, student_logits, reference_logits):
        reference_probs = F.softmax(reference_logits.float(), dim=-1)
        reference_conf, reference_label = reference_probs.max(dim=-1)
        student_probs = F.softmax(student_logits.detach().float(), dim=-1)
        student_conf, student_label = student_probs.max(dim=-1)

        mask = reference_conf.ge(self.pl_threshold)
        if self.pl_use_student_low_conf_mask:
            mask = mask & student_conf.lt(self.pl_student_threshold)

        mask_float = mask.float()
        ce = F.cross_entropy(student_logits, reference_label, reduction="none")
        loss = (ce * mask_float).sum() / mask_float.sum().clamp_min(1.0)
        self._ensure_finite("maple_pseudo_label_loss", loss)

        stats = {
            "coverage": mask_float.mean(),
            "clip_conf": reference_conf.mean(),
            "student_conf": student_conf.mean(),
            "agreement": (student_label == reference_label).float().mean(),
        }
        return loss, stats

    def _dual_view_pseudo_label_terms(
        self,
        strong_logits,
        student_weak_logits,
        reference_weak_logits,
        true_label=None,
    ):
        """Build detached hard/soft targets on weak views and score strong views."""
        teacher_probs = F.softmax(reference_weak_logits.detach().float(), dim=-1)
        student_probs = F.softmax(student_weak_logits.detach().float(), dim=-1)
        teacher_conf, teacher_label = teacher_probs.max(dim=-1)
        student_conf, student_label = student_probs.max(dim=-1)

        high_conf = teacher_conf.ge(self.pl_dual_conf_threshold) & student_conf.ge(
            self.pl_dual_conf_threshold
        )
        agreement = teacher_label.eq(student_label)
        hard_mask = high_conf & agreement
        if self.pl_variant in {
            "agreement_hard_student_soft",
            "agreement_hard_student_top1",
        }:
            soft_mask = student_conf.ge(self.pl_dual_conf_threshold) & teacher_conf.lt(
                self.pl_dual_conf_threshold
            )
            soft_target = student_probs.detach()
            soft_weight = (
                (student_conf - self.pl_dual_conf_threshold)
                / max(1.0 - self.pl_dual_conf_threshold, 1e-12)
            ).clamp_(0.0, 1.0)
            soft_label = student_label
            if self.pl_variant == "agreement_hard_student_soft":
                soft_per_sample = F.kl_div(
                    F.log_softmax(strong_logits.float(), dim=-1),
                    soft_target,
                    reduction="none",
                ).sum(dim=-1)
            else:
                # Mechanism control: keep the student-high/teacher-low mask,
                # confidence weights, and target-batch denominator unchanged,
                # but discard the student's non-top-1 probability mass.
                soft_per_sample = F.cross_entropy(
                    strong_logits.float(), student_label, reduction="none"
                )
        else:
            soft_mask = high_conf & ~agreement
            soft_target = (0.5 * (teacher_probs + student_probs)).detach()
            soft_weight = torch.ones_like(student_conf)
            soft_per_sample = -(
                soft_target * F.log_softmax(strong_logits.float(), dim=-1)
            ).sum(dim=-1)
            soft_label = soft_target.argmax(dim=-1)
        if self.pl_variant == "agreement_hard":
            soft_mask = torch.zeros_like(soft_mask)

        hard_per_sample = F.cross_entropy(
            strong_logits.float(), teacher_label, reduction="none"
        )
        hard_sum = (hard_per_sample * hard_mask.float()).sum()
        selected_soft_weight = soft_weight * soft_mask.float()
        soft_sum = (soft_per_sample * selected_soft_weight).sum()

        # Counterfactual coverage under the old one-teacher PL gate. This does
        # not affect optimization and makes comparisons with legacy runs direct.
        old_mask = teacher_conf.ge(self.pl_threshold)
        if self.pl_use_student_low_conf_mask:
            old_mask = old_mask & student_conf.lt(self.pl_student_threshold)

        soft_entropy = -(soft_target * (soft_target + 1e-12).log()).sum(dim=-1)
        result = {
            "hard_sum": hard_sum,
            "soft_sum": soft_sum,
            "hard_count": hard_mask.float().sum(),
            "soft_count": soft_mask.float().sum(),
            "soft_weight_sum": selected_soft_weight.sum(),
            "high_count": high_conf.float().sum(),
            "old_count": old_mask.float().sum(),
            "total_count": strong_logits.new_tensor(float(strong_logits.shape[0])),
            "teacher_conf_sum": teacher_conf.sum(),
            "student_conf_sum": student_conf.sum(),
            "agreement_count": agreement.float().sum(),
            "soft_entropy_sum": (soft_entropy * soft_mask.float()).sum(),
        }
        if true_label is not None:
            result.update(
                {
                    "hard_correct_count": (
                        teacher_label.eq(true_label) & hard_mask
                    ).float().sum(),
                    "soft_correct_count": (
                        soft_label.eq(true_label) & soft_mask
                    ).float().sum(),
                    "soft_weighted_correct_sum": (
                        soft_label.eq(true_label).float() * selected_soft_weight
                    ).sum(),
                }
            )
        return result

    def _forward_train_dual_pl(self, image_s, label_s, image_u_dict):
        logits_s = self(image_s)
        loss_ce = F.cross_entropy(logits_s, label_s)
        self._ensure_finite("maple_source_ce", loss_ce)
        lambda_pl_current = self.current_lambda_pl()

        terms = []
        for views in image_u_dict.values():
            if not isinstance(views, dict) or not {"weak", "strong"}.issubset(views):
                raise TypeError(
                    "Dual-view PL expects each target batch to contain weak/strong views"
                )
            weak_image = views["weak"]
            strong_logits = self(views["strong"])
            with torch.no_grad():
                student_weak_logits = self(weak_image)
                reference_weak_logits = self._compute_reference_logits(weak_image)
            terms.append(
                self._dual_view_pseudo_label_terms(
                    strong_logits,
                    student_weak_logits,
                    reference_weak_logits,
                    true_label=views.get("true_label"),
                )
            )

        def total(key):
            return torch.stack([item[key] for item in terms]).sum()

        hard_count = total("hard_count")
        soft_count = total("soft_count")
        soft_weight_sum = total("soft_weight_sum")
        total_count = total("total_count").clamp_min(1.0)
        loss_hard = total("hard_sum") / hard_count.clamp_min(1.0)
        if self.pl_variant in {
            "agreement_hard_student_soft",
            "agreement_hard_student_top1",
        }:
            # The fixed target-batch denominator makes a rare soft candidate's
            # contribution proportional to its confidence weight. No selected-
            # count division (and therefore no divide-by-zero branch) is needed.
            loss_soft = total("soft_sum") / total_count
            weighted_soft = self.pl_student_soft_lambda * loss_soft
            loss_pl = loss_hard + loss_soft
        else:
            loss_soft = total("soft_sum") / soft_count.clamp_min(1.0)
            weighted_soft = lambda_pl_current * self.pl_soft_beta * loss_soft
            loss_pl = loss_hard + self.pl_soft_beta * loss_soft
        weighted_hard = lambda_pl_current * loss_hard
        loss_total = loss_ce + weighted_hard + weighted_soft
        self._ensure_finite("maple_dual_pl_total", loss_total)

        selected_count = hard_count + soft_count
        outputs = {
            "loss": loss_total,
            "source_ce": loss_ce.detach(),
            "loss_pl": loss_pl.detach(),
            "loss_pl_hard": loss_hard.detach(),
            "loss_pl_soft": loss_soft.detach(),
            "weighted_loss_pl": (weighted_hard + weighted_soft).detach(),
            "weighted_loss_pl_hard": weighted_hard.detach(),
            "weighted_loss_pl_soft": weighted_soft.detach(),
            "lambda_pl_current": loss_ce.new_tensor(lambda_pl_current),
            "pl_student_soft_lambda": loss_ce.new_tensor(
                self.pl_student_soft_lambda
                if self.pl_variant
                in {
                    "agreement_hard_student_soft",
                    "agreement_hard_student_top1",
                }
                else 0.0
            ),
            "pl_hard_count": hard_count.detach(),
            "pl_soft_count": soft_count.detach(),
            "pl_soft_weight_sum": soft_weight_sum.detach(),
            "pl_selected_count": selected_count.detach(),
            "pl_total_count": total_count.detach(),
            "pl_hard_coverage": (hard_count / total_count).detach(),
            "pl_soft_coverage": (soft_count / total_count).detach(),
            "pl_soft_effective_coverage": (
                soft_weight_sum / total_count
            ).detach(),
            "pl_soft_mean_weight": (
                soft_weight_sum / soft_count.clamp_min(1.0)
            ).detach(),
            "pl_coverage": (selected_count / total_count).detach(),
            "pl_dual_high_coverage": (total("high_count") / total_count).detach(),
            "pl_ignored_coverage": (1.0 - selected_count / total_count).detach(),
            "pl_original_counterfactual_coverage": (
                total("old_count") / total_count
            ).detach(),
            "pl_clip_conf": (total("teacher_conf_sum") / total_count).detach(),
            "pl_student_conf": (total("student_conf_sum") / total_count).detach(),
            "clip_student_agreement": (
                total("agreement_count") / total_count
            ).detach(),
            "pl_soft_target_entropy": (
                total("soft_entropy_sum") / soft_count.clamp_min(1.0)
            ).detach(),
            "pl_hard_active": hard_count.gt(0).float().detach(),
            "pl_soft_active": soft_count.gt(0).float().detach(),
            "_source_ce_objective": loss_ce,
            "_pl_hard_objective": weighted_hard,
            "_pl_soft_objective": weighted_soft,
        }
        if "hard_correct_count" in terms[0]:
            outputs.update(
                {
                    "pl_hard_true_accuracy": (
                        total("hard_correct_count") / hard_count.clamp_min(1.0)
                    ).detach(),
                    "pl_hard_correct_count": total("hard_correct_count").detach(),
                    "pl_soft_argmax_true_accuracy": (
                        total("soft_correct_count") / soft_count.clamp_min(1.0)
                    ).detach(),
                    "pl_student_soft_true_accuracy": (
                        total("soft_correct_count") / soft_count.clamp_min(1.0)
                    ).detach(),
                    "pl_soft_correct_count": total("soft_correct_count").detach(),
                    "pl_soft_weighted_true_accuracy": (
                        total("soft_weighted_correct_sum")
                        / soft_weight_sum.clamp_min(1.0)
                    ).detach(),
                    "pl_soft_weighted_correct_sum": total(
                        "soft_weighted_correct_sum"
                    ).detach(),
                }
            )
        return outputs

    @torch.no_grad()
    def _update_weak_pl_state(self, domain_name, reference_probs):
        device = reference_probs.device
        num_classes = reference_probs.shape[-1]
        reference_conf, reference_label = reference_probs.max(dim=-1)
        entropy = -(
            reference_probs * (reference_probs + self.weak_pl_eps).log()
        ).sum(dim=-1)

        count = torch.bincount(reference_label, minlength=num_classes).float()
        count_dist = count / count.sum().clamp_min(1.0)
        entropy_sum = torch.zeros(num_classes, device=device)
        entropy_sum.scatter_add_(0, reference_label, entropy)
        observed = count > 0
        entropy_avg = torch.full(
            (num_classes,),
            float(entropy.mean().detach().item()),
            device=device,
            dtype=entropy.dtype,
        )
        entropy_avg[observed] = entropy_sum[observed] / count[observed]

        if domain_name not in self._weak_pl_domain_state:
            self._weak_pl_domain_state[domain_name] = {
                "count": count_dist.detach(),
                "entropy": entropy_avg.detach(),
                "seen": observed.detach().clone(),
            }
        else:
            state = self._weak_pl_domain_state[domain_name]
            momentum = self.weak_pl_momentum
            state["count"] = (
                momentum * state["count"].to(device)
                + (1.0 - momentum) * count_dist.detach()
            )
            old_entropy = state["entropy"].to(device)
            new_entropy = old_entropy.clone()
            new_entropy[observed] = (
                momentum * old_entropy[observed]
                + (1.0 - momentum) * entropy_avg[observed].detach()
            )
            state["entropy"] = new_entropy
            state["seen"] = state["seen"].to(device) | observed

        state = self._weak_pl_domain_state[domain_name]
        count_ema = state["count"].to(device)
        entropy_ema = state["entropy"].to(device)
        seen = state["seen"].to(device)

        count_order = torch.argsort(count_ema, descending=False)
        entropy_order = torch.argsort(entropy_ema, descending=True)
        rank_values = torch.arange(
            num_classes, 0, -1, device=device, dtype=count_ema.dtype
        )
        count_rank = torch.empty_like(count_ema)
        entropy_rank = torch.empty_like(entropy_ema)
        count_rank[count_order] = rank_values
        entropy_rank[entropy_order] = rank_values
        weak_score = count_rank + entropy_rank
        weak_score = torch.where(
            seen,
            weak_score,
            torch.full_like(weak_score, -float("inf")),
        )

        topk = max(1, int(round(self.weak_pl_fraction * num_classes)))
        topk = min(int(seen.float().sum().item()), topk)
        weak_class_mask = torch.zeros(num_classes, device=device, dtype=torch.bool)
        if topk > 0:
            weak_indices = torch.topk(weak_score, k=topk, largest=True).indices
            weak_class_mask[weak_indices] = True

        weak_stats = {
            "weak_class_count": weak_class_mask.float().sum(),
            "weak_class_fraction": weak_class_mask.float().mean(),
            "weak_class_count_ema": self._masked_mean(count_ema, weak_class_mask),
            "weak_class_entropy_ema": self._masked_mean(
                entropy_ema, weak_class_mask
            ),
            "all_class_entropy_ema": entropy_ema.mean(),
            "all_class_count_ema": count_ema.mean(),
        }
        return weak_class_mask, weak_stats

    def _weak_pseudo_label_loss(self, domain_name, student_logits, reference_logits):
        reference_probs = F.softmax(reference_logits.float(), dim=-1)
        reference_conf, reference_label = reference_probs.max(dim=-1)
        student_probs = F.softmax(student_logits.detach().float(), dim=-1)
        student_conf, student_label = student_probs.max(dim=-1)
        weak_class_mask, weak_stats = self._update_weak_pl_state(
            domain_name, reference_probs.detach()
        )

        weak_candidate_mask = weak_class_mask[reference_label]
        mask = weak_candidate_mask
        mask = mask & reference_conf.ge(self.weak_pl_teacher_threshold)
        mask = mask & reference_conf.lt(self.weak_pl_teacher_threshold_high)
        if self.weak_pl_use_student_low_conf_mask:
            mask = mask & student_conf.lt(self.weak_pl_student_threshold)

        mask_float = mask.float()
        if mask_float.sum().item() > 0:
            ce = F.cross_entropy(student_logits, reference_label, reduction="none")
            loss = (ce * mask_float).sum() / mask_float.sum().clamp_min(1.0)
        else:
            loss = student_logits.new_zeros(())
        self._ensure_finite("maple_weak_pseudo_label_loss", loss)

        stats = {
            "coverage": mask_float.mean(),
            "candidate_coverage": weak_candidate_mask.float().mean(),
            "clip_conf": reference_conf.mean(),
            "student_conf": student_conf.mean(),
            "selected_clip_conf": self._masked_mean(reference_conf, mask),
            "selected_student_conf": self._masked_mean(student_conf, mask),
            "candidate_clip_conf": self._masked_mean(
                reference_conf, weak_candidate_mask
            ),
            "candidate_student_conf": self._masked_mean(
                student_conf, weak_candidate_mask
            ),
            "agreement": (student_label == reference_label).float().mean(),
            "selected_agreement": self._masked_mean(
                (student_label == reference_label).float(), mask
            ),
            "weight": reference_conf.new_tensor(self.lambda_weak_pl),
            "teacher_threshold": reference_conf.new_tensor(
                self.weak_pl_teacher_threshold
            ),
            "teacher_threshold_high": reference_conf.new_tensor(
                self.weak_pl_teacher_threshold_high
            ),
            "student_threshold": reference_conf.new_tensor(
                self.weak_pl_student_threshold
            ),
        }
        stats.update(weak_stats)
        return loss, stats

    def _self_distill_loss(
        self, domain_name, student_logits, image, reference_logits=None
    ):
        del domain_name
        old_model = self._self_distill_old_model_holder[0]
        if old_model is None:
            raise RuntimeError("SELF_DISTILL is enabled but old student is not built")

        with torch.no_grad():
            old_logits = old_model(image)
            old_probs = F.softmax(old_logits.float(), dim=-1)
            old_conf, old_label = old_probs.max(dim=-1)

        student_probs = F.softmax(student_logits.detach().float(), dim=-1)
        student_conf, student_label = student_probs.max(dim=-1)
        reference_conf = old_conf.new_zeros(old_conf.shape)
        reference_label = old_label.clone()
        if reference_logits is not None:
            reference_probs = F.softmax(reference_logits.float(), dim=-1)
            reference_conf, reference_label = reference_probs.max(dim=-1)

        mask = build_self_distill_mask(
            self.self_distill_mode,
            old_conf,
            old_conf_low=self.self_distill_old_conf_low,
            old_conf_high=self.self_distill_old_conf_high,
            reference_conf=reference_conf if reference_logits is not None else None,
            clip_conf_high=self.self_distill_clip_conf_high,
        )
        mask_float = mask.float()

        temperature = max(self.self_distill_temperature, self.self_distill_eps)
        old_soft = F.softmax(old_logits.float() / temperature, dim=-1)
        student_log_soft = F.log_softmax(student_logits.float() / temperature, dim=-1)
        kl_per_sample = (
            F.kl_div(student_log_soft, old_soft, reduction="none").sum(dim=-1)
            * temperature
            * temperature
        )
        loss = (kl_per_sample * mask_float).sum() / mask_float.sum().clamp_min(1.0)
        self._ensure_finite("maple_self_distill_loss", loss)

        old_entropy = -(old_probs * (old_probs + self.self_distill_eps).log()).sum(
            dim=-1
        )
        stats = {
            "coverage": mask_float.mean(),
            "old_conf": old_conf.mean(),
            "student_conf": student_conf.mean(),
            "selected_old_conf": self._masked_mean(old_conf, mask),
            "clip_conf": reference_conf.mean(),
            "selected_clip_conf": self._masked_mean(reference_conf, mask),
            "old_clip_agreement": (old_label == reference_label).float().mean(),
            "selected_old_clip_agreement": self._masked_mean(
                (old_label == reference_label).float(), mask
            ),
            "selected_student_conf": self._masked_mean(student_conf, mask),
            "kl": kl_per_sample.mean(),
            "selected_kl": self._masked_mean(kl_per_sample, mask),
            "old_entropy": old_entropy.mean(),
            "selected_old_entropy": self._masked_mean(old_entropy, mask),
            "agreement": (student_label == old_label).float().mean(),
            "selected_agreement": self._masked_mean(
                (student_label == old_label).float(), mask
            ),
            "weight": student_logits.new_tensor(self.lambda_self_distill),
            "temperature": student_logits.new_tensor(self.self_distill_temperature),
            "old_conf_low": student_logits.new_tensor(
                self.self_distill_old_conf_low
            ),
            "old_conf_high": student_logits.new_tensor(
                self.self_distill_old_conf_high
            ),
        }
        return loss, stats

    def forward_train(self, image_s, label_s, image_u_dict):
        if self.lambda_ent > 0.0:
            logits_s = self(image_s)
            loss_ce = F.cross_entropy(logits_s, label_s)
            if list(image_u_dict) != ["mixed_target"]:
                raise RuntimeError(
                    f"Expected one mixed target batch, got {list(image_u_dict)}"
                )
            target_logits = self(image_u_dict["mixed_target"])
            probabilities = F.softmax(target_logits.float(), dim=-1)
            loss_ent = -(
                probabilities * probabilities.clamp_min(self.entropy_eps).log()
            ).sum(dim=-1).mean()
            loss_total = loss_ce + self.lambda_ent * loss_ent
            self._ensure_finite("maple_mixed_target_entropy_total", loss_total)
            return {
                "loss": loss_total,
                "source_ce": loss_ce.detach(),
                "target_entropy": loss_ent.detach(),
                "entropy_weight": loss_ce.new_tensor(self.lambda_ent),
            }
        if self.pl_variant != "legacy":
            return self._forward_train_dual_pl(image_s, label_s, image_u_dict)
        logits_s = self(image_s)
        loss_ce = F.cross_entropy(logits_s, label_s)
        self._ensure_finite("maple_source_ce", loss_ce)

        pl_by_domain = {}
        pl_stats_by_domain = {}
        weak_pl_by_domain = {}
        weak_pl_stats_by_domain = {}
        self_distill_by_domain = {}
        self_distill_stats_by_domain = {}
        lambda_pl_current = self.current_lambda_pl()
        use_clean_pl = self.lambda_pl > 0.0
        use_weak_pl = self.weak_pl_enabled and self.lambda_weak_pl > 0.0
        use_self_distill = (
            self.self_distill_enabled and self.lambda_self_distill > 0.0
        )
        handoff_needs_reference = (
            use_self_distill and self.self_distill_mode == "teacher_handoff"
        )
        if use_clean_pl or use_weak_pl or use_self_distill:
            for domain_name, image_u in image_u_dict.items():
                target_logits = self(image_u)
                reference_logits = None
                if use_clean_pl or use_weak_pl or handoff_needs_reference:
                    reference_logits = self._compute_reference_logits(image_u)
                    if use_clean_pl:
                        pl_loss, pl_stats = self._pseudo_label_loss(
                            target_logits, reference_logits
                        )
                        pl_by_domain[domain_name] = pl_loss
                        pl_stats_by_domain[domain_name] = pl_stats
                    if use_weak_pl:
                        weak_pl_loss, weak_pl_stats = self._weak_pseudo_label_loss(
                            domain_name, target_logits, reference_logits
                        )
                        weak_pl_by_domain[domain_name] = weak_pl_loss
                        weak_pl_stats_by_domain[domain_name] = weak_pl_stats
                if use_self_distill:
                    sd_loss, sd_stats = self._self_distill_loss(
                        domain_name,
                        target_logits,
                        image_u,
                        reference_logits=reference_logits,
                    )
                    self_distill_by_domain[domain_name] = sd_loss
                    self_distill_stats_by_domain[domain_name] = sd_stats

        if use_clean_pl:
            loss_pl = torch.stack(list(pl_by_domain.values())).mean()
            pl_coverage = torch.stack(
                [stats["coverage"] for stats in pl_stats_by_domain.values()]
            ).mean()
            pl_clip_conf = torch.stack(
                [stats["clip_conf"] for stats in pl_stats_by_domain.values()]
            ).mean()
            pl_student_conf = torch.stack(
                [stats["student_conf"] for stats in pl_stats_by_domain.values()]
            ).mean()
            clip_student_agreement = torch.stack(
                [stats["agreement"] for stats in pl_stats_by_domain.values()]
            ).mean()
        else:
            loss_pl = loss_ce.new_zeros(())
            pl_coverage = loss_ce.new_zeros(())
            pl_clip_conf = loss_ce.new_zeros(())
            pl_student_conf = loss_ce.new_zeros(())
            clip_student_agreement = loss_ce.new_zeros(())

        if use_weak_pl:
            weak_loss_pl = torch.stack(list(weak_pl_by_domain.values())).mean()
            weak_pl_coverage = torch.stack(
                [stats["coverage"] for stats in weak_pl_stats_by_domain.values()]
            ).mean()
            weak_pl_candidate_coverage = torch.stack(
                [
                    stats["candidate_coverage"]
                    for stats in weak_pl_stats_by_domain.values()
                ]
            ).mean()
            weak_pl_clip_conf = torch.stack(
                [
                    stats["selected_clip_conf"]
                    for stats in weak_pl_stats_by_domain.values()
                ]
            ).mean()
            weak_pl_student_conf = torch.stack(
                [
                    stats["selected_student_conf"]
                    for stats in weak_pl_stats_by_domain.values()
                ]
            ).mean()
            weak_pl_candidate_clip_conf = torch.stack(
                [
                    stats["candidate_clip_conf"]
                    for stats in weak_pl_stats_by_domain.values()
                ]
            ).mean()
            weak_pl_candidate_student_conf = torch.stack(
                [
                    stats["candidate_student_conf"]
                    for stats in weak_pl_stats_by_domain.values()
                ]
            ).mean()
            weak_pl_selected_agreement = torch.stack(
                [
                    stats["selected_agreement"]
                    for stats in weak_pl_stats_by_domain.values()
                ]
            ).mean()
            weak_pl_class_count = torch.stack(
                [
                    stats["weak_class_count"]
                    for stats in weak_pl_stats_by_domain.values()
                ]
            ).mean()
            weak_pl_class_count_ema = torch.stack(
                [
                    stats["weak_class_count_ema"]
                    for stats in weak_pl_stats_by_domain.values()
                ]
            ).mean()
            weak_pl_class_entropy_ema = torch.stack(
                [
                    stats["weak_class_entropy_ema"]
                    for stats in weak_pl_stats_by_domain.values()
                ]
            ).mean()
        else:
            weak_loss_pl = loss_ce.new_zeros(())
            weak_pl_coverage = loss_ce.new_zeros(())
            weak_pl_candidate_coverage = loss_ce.new_zeros(())
            weak_pl_clip_conf = loss_ce.new_zeros(())
            weak_pl_student_conf = loss_ce.new_zeros(())
            weak_pl_candidate_clip_conf = loss_ce.new_zeros(())
            weak_pl_candidate_student_conf = loss_ce.new_zeros(())
            weak_pl_selected_agreement = loss_ce.new_zeros(())
            weak_pl_class_count = loss_ce.new_zeros(())
            weak_pl_class_count_ema = loss_ce.new_zeros(())
            weak_pl_class_entropy_ema = loss_ce.new_zeros(())

        if use_self_distill:
            loss_self_distill = torch.stack(
                list(self_distill_by_domain.values())
            ).mean()
            self_distill_coverage = torch.stack(
                [
                    stats["coverage"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_old_conf = torch.stack(
                [
                    stats["old_conf"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_student_conf = torch.stack(
                [
                    stats["student_conf"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_selected_old_conf = torch.stack(
                [
                    stats["selected_old_conf"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_selected_clip_conf = torch.stack(
                [
                    stats["selected_clip_conf"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_selected_old_clip_agreement = torch.stack(
                [
                    stats["selected_old_clip_agreement"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_selected_student_conf = torch.stack(
                [
                    stats["selected_student_conf"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_kl = torch.stack(
                [stats["kl"] for stats in self_distill_stats_by_domain.values()]
            ).mean()
            self_distill_selected_kl = torch.stack(
                [
                    stats["selected_kl"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_old_entropy = torch.stack(
                [
                    stats["old_entropy"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_selected_old_entropy = torch.stack(
                [
                    stats["selected_old_entropy"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_agreement = torch.stack(
                [
                    stats["agreement"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
            self_distill_selected_agreement = torch.stack(
                [
                    stats["selected_agreement"]
                    for stats in self_distill_stats_by_domain.values()
                ]
            ).mean()
        else:
            loss_self_distill = loss_ce.new_zeros(())
            self_distill_coverage = loss_ce.new_zeros(())
            self_distill_old_conf = loss_ce.new_zeros(())
            self_distill_student_conf = loss_ce.new_zeros(())
            self_distill_selected_old_conf = loss_ce.new_zeros(())
            self_distill_selected_clip_conf = loss_ce.new_zeros(())
            self_distill_selected_old_clip_agreement = loss_ce.new_zeros(())
            self_distill_selected_student_conf = loss_ce.new_zeros(())
            self_distill_kl = loss_ce.new_zeros(())
            self_distill_selected_kl = loss_ce.new_zeros(())
            self_distill_old_entropy = loss_ce.new_zeros(())
            self_distill_selected_old_entropy = loss_ce.new_zeros(())
            self_distill_agreement = loss_ce.new_zeros(())
            self_distill_selected_agreement = loss_ce.new_zeros(())

        loss_total = (
            loss_ce
            + lambda_pl_current * loss_pl
            + self.lambda_weak_pl * weak_loss_pl
            + self.lambda_self_distill * loss_self_distill
        )
        self._ensure_finite("maple_loss_total", loss_total)

        if self.debug_print_once and not self._debug_printed:
            print("[MaPLeMTDA PL debug]")
            print("source batch shape:", tuple(image_s.shape))
            for domain_name, image_u in image_u_dict.items():
                print(f"target batch shape [{domain_name}]:", tuple(image_u.shape))
            print("source logits shape:", tuple(logits_s.shape))
            print("lambda_pl:", lambda_pl_current)
            print("loss_ce:", float(loss_ce.detach().item()))
            print("loss_pl:", float(loss_pl.detach().item()))
            print("pl_coverage:", float(pl_coverage.detach().item()))
            print("weak_loss_pl:", float(weak_loss_pl.detach().item()))
            print("weak_pl_coverage:", float(weak_pl_coverage.detach().item()))
            print(
                "weak_pl_class_count:",
                float(weak_pl_class_count.detach().item()),
            )
            print("loss_self_distill:", float(loss_self_distill.detach().item()))
            print(
                "self_distill_coverage:",
                float(self_distill_coverage.detach().item()),
            )
            print("loss_total:", float(loss_total.detach().item()))
            self._debug_printed = True

        outputs = {
            "loss": loss_total,
            "source_ce": loss_ce.detach(),
            "loss_pl": loss_pl.detach(),
            "weighted_loss_pl": (lambda_pl_current * loss_pl).detach(),
            "lambda_pl_current": loss_ce.new_tensor(lambda_pl_current),
            "pl_coverage": pl_coverage.detach(),
            "pl_clip_conf": pl_clip_conf.detach(),
            "pl_student_conf": pl_student_conf.detach(),
            "clip_student_agreement": clip_student_agreement.detach(),
        }
        if use_self_distill:
            outputs.update(
                {
                    "loss_self_distill": loss_self_distill.detach(),
                    "weighted_loss_self_distill": (
                        self.lambda_self_distill * loss_self_distill
                    ).detach(),
                    "self_distill_coverage": self_distill_coverage.detach(),
                    "self_distill_old_conf": self_distill_old_conf.detach(),
                    "self_distill_student_conf": (
                        self_distill_student_conf.detach()
                    ),
                    "self_distill_selected_old_conf": (
                        self_distill_selected_old_conf.detach()
                    ),
                    "self_distill_selected_clip_conf": (
                        self_distill_selected_clip_conf.detach()
                    ),
                    "self_distill_selected_old_clip_agreement": (
                        self_distill_selected_old_clip_agreement.detach()
                    ),
                    "self_distill_selected_student_conf": (
                        self_distill_selected_student_conf.detach()
                    ),
                    "self_distill_kl": self_distill_kl.detach(),
                    "self_distill_selected_kl": self_distill_selected_kl.detach(),
                    "self_distill_old_entropy": self_distill_old_entropy.detach(),
                    "self_distill_selected_old_entropy": (
                        self_distill_selected_old_entropy.detach()
                    ),
                    "self_distill_agreement": self_distill_agreement.detach(),
                    "self_distill_selected_agreement": (
                        self_distill_selected_agreement.detach()
                    ),
                    "self_distill_weight": loss_ce.new_tensor(
                        self.lambda_self_distill
                    ).detach(),
                    "self_distill_temperature": loss_ce.new_tensor(
                        self.self_distill_temperature
                    ).detach(),
                    "self_distill_old_conf_low": loss_ce.new_tensor(
                        self.self_distill_old_conf_low
                    ).detach(),
                    "self_distill_old_conf_high": loss_ce.new_tensor(
                        self.self_distill_old_conf_high
                    ).detach(),
                }
            )
        if use_weak_pl:
            outputs.update(
                {
                    "weak_loss_pl": weak_loss_pl.detach(),
                    "weighted_weak_loss_pl": (
                        self.lambda_weak_pl * weak_loss_pl
                    ).detach(),
                    "weak_pl_coverage": weak_pl_coverage.detach(),
                    "weak_pl_candidate_coverage": (
                        weak_pl_candidate_coverage.detach()
                    ),
                    "weak_pl_clip_conf": weak_pl_clip_conf.detach(),
                    "weak_pl_student_conf": weak_pl_student_conf.detach(),
                    "weak_pl_candidate_clip_conf": (
                        weak_pl_candidate_clip_conf.detach()
                    ),
                    "weak_pl_candidate_student_conf": (
                        weak_pl_candidate_student_conf.detach()
                    ),
                    "weak_pl_selected_agreement": (
                        weak_pl_selected_agreement.detach()
                    ),
                    "weak_pl_class_count": weak_pl_class_count.detach(),
                    "weak_pl_class_count_ema": weak_pl_class_count_ema.detach(),
                    "weak_pl_class_entropy_ema": (
                        weak_pl_class_entropy_ema.detach()
                    ),
                    "weak_pl_weight": loss_ce.new_tensor(
                        self.lambda_weak_pl
                    ).detach(),
                    "weak_pl_teacher_threshold": loss_ce.new_tensor(
                        self.weak_pl_teacher_threshold
                    ).detach(),
                    "weak_pl_teacher_threshold_high": loss_ce.new_tensor(
                        self.weak_pl_teacher_threshold_high
                    ).detach(),
                    "weak_pl_student_threshold": loss_ce.new_tensor(
                        self.weak_pl_student_threshold
                    ).detach(),
                }
            )
        for domain_name, pl_loss in pl_by_domain.items():
            outputs[f"loss_pl_{domain_name}"] = pl_loss.detach()
        for domain_name, stats in pl_stats_by_domain.items():
            outputs[f"pl_coverage_{domain_name}"] = stats["coverage"].detach()
            outputs[f"pl_clip_conf_{domain_name}"] = stats["clip_conf"].detach()
            outputs[f"pl_student_conf_{domain_name}"] = stats[
                "student_conf"
            ].detach()
            outputs[f"clip_student_agreement_{domain_name}"] = stats[
                "agreement"
            ].detach()
        for domain_name, sd_loss in self_distill_by_domain.items():
            outputs[f"loss_self_distill_{domain_name}"] = sd_loss.detach()
        for domain_name, stats in self_distill_stats_by_domain.items():
            outputs[f"self_distill_coverage_{domain_name}"] = stats[
                "coverage"
            ].detach()
            outputs[f"self_distill_old_conf_{domain_name}"] = stats[
                "old_conf"
            ].detach()
            outputs[f"self_distill_student_conf_{domain_name}"] = stats[
                "student_conf"
            ].detach()
            outputs[f"self_distill_selected_old_conf_{domain_name}"] = stats[
                "selected_old_conf"
            ].detach()
            outputs[f"self_distill_selected_clip_conf_{domain_name}"] = stats[
                "selected_clip_conf"
            ].detach()
            outputs[f"self_distill_selected_old_clip_agreement_{domain_name}"] = stats[
                "selected_old_clip_agreement"
            ].detach()
            outputs[f"self_distill_selected_student_conf_{domain_name}"] = stats[
                "selected_student_conf"
            ].detach()
            outputs[f"self_distill_selected_kl_{domain_name}"] = stats[
                "selected_kl"
            ].detach()
            outputs[f"self_distill_selected_agreement_{domain_name}"] = stats[
                "selected_agreement"
            ].detach()
        for domain_name, weak_pl_loss in weak_pl_by_domain.items():
            outputs[f"weak_loss_pl_{domain_name}"] = weak_pl_loss.detach()
        for domain_name, stats in weak_pl_stats_by_domain.items():
            outputs[f"weak_pl_coverage_{domain_name}"] = stats[
                "coverage"
            ].detach()
            outputs[f"weak_pl_candidate_coverage_{domain_name}"] = stats[
                "candidate_coverage"
            ].detach()
            outputs[f"weak_pl_clip_conf_{domain_name}"] = stats[
                "selected_clip_conf"
            ].detach()
            outputs[f"weak_pl_student_conf_{domain_name}"] = stats[
                "selected_student_conf"
            ].detach()
            outputs[f"weak_pl_candidate_clip_conf_{domain_name}"] = stats[
                "candidate_clip_conf"
            ].detach()
            outputs[f"weak_pl_candidate_student_conf_{domain_name}"] = stats[
                "candidate_student_conf"
            ].detach()
            outputs[f"weak_pl_selected_agreement_{domain_name}"] = stats[
                "selected_agreement"
            ].detach()
            outputs[f"weak_pl_class_count_{domain_name}"] = stats[
                "weak_class_count"
            ].detach()
            outputs[f"weak_pl_class_count_ema_{domain_name}"] = stats[
                "weak_class_count_ema"
            ].detach()
            outputs[f"weak_pl_class_entropy_ema_{domain_name}"] = stats[
                "weak_class_entropy_ema"
            ].detach()
        return outputs


@TRAINER_REGISTRY.register()
class MaPLeMTDA(MultiTargetTrainerXU):
    """MaPLe source-only baseline under the Office-Home SS-MTDA protocol."""

    def check_cfg(self, cfg):
        maple_cfg = cfg.TRAINER.MAPLE_MTDA
        assert maple_cfg.PREC in ["fp16", "fp32", "amp"]
        assert maple_cfg.LAMBDA_PL >= 0.0
        assert float(cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT) >= 0.0
        assert maple_cfg.LAMBDA_PL_FINAL >= 0.0
        assert maple_cfg.PL_SCHEDULE in ["constant", "cosine"]
        assert 0.0 <= maple_cfg.PL_THRESHOLD <= 1.0
        assert 0.0 <= maple_cfg.PL_STUDENT_THRESHOLD <= 1.0
        assert str(maple_cfg.PL_VARIANT).lower() in {
            "legacy",
            "agreement_hard",
            "agreement_hard_soft",
            "agreement_hard_student_soft",
            "agreement_hard_student_top1",
        }
        assert 0.0 <= float(maple_cfg.PL_DUAL_CONF_THRESHOLD) <= 1.0
        assert float(maple_cfg.PL_SOFT_BETA) >= 0.0
        assert float(maple_cfg.PL_STUDENT_SOFT_LAMBDA) >= 0.0
        assert int(maple_cfg.PL_DIAGNOSTIC_WINDOW) > 0
        assert int(maple_cfg.PL_GRAD_AUDIT_INTERVAL) > 0
        assert 0.0 <= maple_cfg.SELF_DISTILL.OLD_CONF_LOW <= 1.0
        assert 0.0 <= maple_cfg.SELF_DISTILL.OLD_CONF_HIGH <= 1.0
        assert maple_cfg.SELF_DISTILL.OLD_CONF_LOW <= maple_cfg.SELF_DISTILL.OLD_CONF_HIGH
        assert maple_cfg.SELF_DISTILL.LAMBDA >= 0.0
        assert maple_cfg.SELF_DISTILL.TEMPERATURE > 0.0
        assert maple_cfg.SELF_DISTILL.MODE in [
            "all",
            "confidence_band",
            "teacher_handoff",
        ]
        assert 0.0 <= maple_cfg.SELF_DISTILL.CLIP_CONF_HIGH <= 1.0
        assert "{}" in maple_cfg.ZS_PROMPT_TEMPLATE
        if str(maple_cfg.PL_VARIANT).lower() != "legacy":
            assert not bool(maple_cfg.WEAK_PL.ENABLED), (
                "Dual-view PL v1 must be isolated from WEAK_PL"
            )
            assert not bool(maple_cfg.SELF_DISTILL.ENABLED), (
                "Dual-view PL v1 must be isolated from SELF_DISTILL"
            )

    def build_data_loader(self):
        maple_cfg = self.cfg.TRAINER.MAPLE_MTDA
        if str(maple_cfg.PL_VARIANT).lower() == "legacy":
            return super().build_data_loader()

        weak_transform = build_transform(self.cfg, is_train=True)
        strong_choice = str(maple_cfg.PL_STRONG_AUGMENT).strip()
        strong_transforms = list(self.cfg.INPUT.TRANSFORMS)
        if strong_choice and strong_choice not in strong_transforms:
            normalize_index = (
                strong_transforms.index("normalize")
                if "normalize" in strong_transforms
                else len(strong_transforms)
            )
            strong_transforms.insert(normalize_index, strong_choice)
        strong_transform = build_transform(
            self.cfg, is_train=True, choices=strong_transforms
        )
        dm = MultiTargetDataManager(
            self.cfg,
            custom_tfm_train_x=weak_transform,
            custom_tfm_train_u=[weak_transform, strong_transform],
        )
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.test_loaders_by_domain = dm.test_loaders_by_domain
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm
        print(f"Dual-view PL weak transforms: {list(self.cfg.INPUT.TRANSFORMS)}")
        print(f"Dual-view PL strong transforms: {strong_transforms}")

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"].to(self.device)
        label_x = batch_x["label"].to(self.device)
        input_u = OrderedDict()
        dual_view = (
            str(self.cfg.TRAINER.MAPLE_MTDA.PL_VARIANT).lower() != "legacy"
        )
        curriculum_cfg = getattr(
            self.cfg.TRAINER.MAPLE_MTDA, "CURRICULUM", None
        )
        diagnostics_enabled = bool(
            curriculum_cfg is not None
            and bool(curriculum_cfg.DIAGNOSTICS.ENABLED)
        )
        for domain_name, batch in batch_u.items():
            if dual_view:
                if "img2" not in batch:
                    raise KeyError("Dual-view PL target batch is missing img2")
                views = {
                    "weak": batch["img"].to(self.device),
                    "strong": batch["img2"].to(self.device),
                }
                if diagnostics_enabled:
                    # Diagnostic only: never consulted by target construction.
                    views["true_label"] = batch["label"].to(self.device)
                input_u[domain_name] = views
            else:
                input_u[domain_name] = batch["img"].to(self.device)
        return input_x, label_x, input_u

    def _registered_model_name(self):
        names = self.get_model_names()
        if len(names) != 1:
            raise RuntimeError(f"Expected one registered model, got {names}")
        return names[0]

    def _maybe_load_post_init_model(self):
        post_cfg = self.cfg.TRAINER.MAPLE_MTDA.POST_INIT
        if not bool(post_cfg.ENABLED):
            return
        if not post_cfg.MODEL_DIR:
            raise ValueError("MAPLE_MTDA.POST_INIT.MODEL_DIR is required when enabled")

        model_name = self._registered_model_name()
        checkpoint_model_name = str(post_cfg.CHECKPOINT_MODEL_NAME).strip()
        if not checkpoint_model_name:
            checkpoint_model_name = model_name
        load_epoch = int(post_cfg.LOAD_EPOCH)
        model_file = "model-best.pth.tar" if load_epoch <= 0 else f"model.pth.tar-{load_epoch}"
        model_path = osp.join(
            str(post_cfg.MODEL_DIR), checkpoint_model_name, model_file
        )
        if not osp.exists(model_path):
            raise FileNotFoundError(f'Post-init model not found at "{model_path}"')

        checkpoint = load_checkpoint_compat(model_path)
        state_dict = checkpoint["state_dict"]
        for key in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
            state_dict.pop(key, None)

        print(
            f'Post-init loading {checkpoint_model_name} weights into '
            f'{model_name} from "{model_path}"'
        )
        load_state_dict_checked(
            self._models[model_name],
            state_dict,
            allowed_missing=(
                "prompt_learner.token_prefix",
                "prompt_learner.token_suffix",
            ),
            context=f"post-init checkpoint {model_path}",
        )

    def _maybe_build_self_distill_old_model(self):
        sd_cfg = self.cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL
        if not (bool(sd_cfg.ENABLED) and float(sd_cfg.LAMBDA) > 0.0):
            return
        print("Building frozen old-student snapshot for self-distillation")
        self.model.build_self_distill_old_model()

    def _finish_maple_post_build_setup(self):
        self._maybe_load_post_init_model()
        self._maybe_build_self_distill_old_model()

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        if cfg.TRAINER.MAPLE_MTDA.USE_MAPLE_CLIP_BUILD:
            clip_model = load_clip_to_cpu(cfg)
        else:
            clip_model = load_base_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE_MTDA.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building MaPLeMTDA custom CLIP")
        self.model = CustomMaPLeMTDA(cfg, classnames, clip_model)
        self.uses_target_training = (
            float(cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT) > 0.0
        )
        if self.uses_target_training:
            assert float(cfg.TRAINER.MAPLE_MTDA.LAMBDA_PL) == 0.0
            assert not bool(cfg.TRAINER.MAPLE_MTDA.WEAK_PL.ENABLED)
            assert not bool(cfg.TRAINER.MAPLE_MTDA.SELF_DISTILL.ENABLED)
        print(
            "MaPLe mixed-target entropy weight: "
            f"{cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT}"
        )

        print("Freezing CLIP image/text encoders; updating MaPLe prompt learner only")
        for name, param in self.model.named_parameters():
            param.requires_grad_("prompt_learner" in name)

        enabled = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        print("Trainable parameters:")
        for name in enabled:
            print(f"  {name}")
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        print(f"Trainable parameter count: {trainable_params:,}")

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MaPLeMTDA", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.MAPLE_MTDA.PREC == "amp" else None
        self._finish_maple_post_build_setup()

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u = self.parse_batch_train(batch_x, batch_u)

        progress = 0.0
        if getattr(self, "max_epoch", 0) > 0 and getattr(self, "num_batches", 0) > 0:
            denom = max(self.max_epoch * self.num_batches - 1, 1)
            progress = (self.epoch * self.num_batches + self.batch_idx) / denom
        if hasattr(self.model, "set_training_progress"):
            self.model.set_training_progress(progress)

        if (
            bool(self.cfg.TRAINER.MAPLE_MTDA.DEBUG.PRINT_ONCE)
            and not getattr(self, "_debug_printed", False)
        ):
            print("MaPLeMTDA debug:")
            print(f"  source domains: {self.cfg.DATASET.SOURCE_DOMAINS}")
            print(f"  target domains: {list(image_u.keys())}")
            print(f"  source batch shape: {tuple(image_x.shape)}")
            for domain_name, image in image_u.items():
                if isinstance(image, dict):
                    print(
                        f"  target batch shapes[{domain_name}]: "
                        f"weak={tuple(image['weak'].shape)}, "
                        f"strong={tuple(image['strong'].shape)}"
                    )
                else:
                    print(f"  target batch shape[{domain_name}]: {tuple(image.shape)}")
            self._debug_printed = True

        prec = self.cfg.TRAINER.MAPLE_MTDA.PREC
        if prec == "amp":
            with autocast():
                outputs = self.model.forward_train(image_x, label_x, image_u)
                loss = outputs["loss"]
            for key in (
                "_source_ce_objective",
                "_pl_hard_objective",
                "_pl_soft_objective",
            ):
                outputs.pop(key, None)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            outputs = self.model.forward_train(image_x, label_x, image_u)
            loss = outputs["loss"]
            for key in (
                "_source_ce_objective",
                "_pl_hard_objective",
                "_pl_soft_objective",
            ):
                outputs.pop(key, None)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_summary = {"loss": float(loss.item())}
        for key, value in outputs.items():
            if key == "loss":
                continue
            loss_summary[key] = value.item() if torch.is_tensor(value) else float(value)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        model_path = osp.join(directory, "MaPLeMTDA", model_file)
        if not osp.exists(model_path):
            raise FileNotFoundError(f'Model not found at "{model_path}"')

        checkpoint = load_checkpoint_compat(model_path)
        state_dict = checkpoint["state_dict"]
        for key in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
            state_dict.pop(key, None)

        print(f'Loading weights to MaPLeMTDA from "{model_path}"')
        load_state_dict_checked(
            self._models["MaPLeMTDA"],
            state_dict,
            allowed_missing=(
                "prompt_learner.token_prefix",
                "prompt_learner.token_suffix",
            ),
            context=f"MaPLeMTDA checkpoint {model_path}",
        )
