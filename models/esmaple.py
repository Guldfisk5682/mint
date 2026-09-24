import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.clip import clip
from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler, build_optimizer

from models.checkpoint_utils import load_checkpoint_compat, load_state_dict_checked
from models.maple import (
    CustomMaPLeMTDA,
    MaPLeMTDA,
    TextEncoderMaPLe,
    _get_clones,
    load_base_clip_to_cpu,
    load_clip_to_cpu,
)


class ContinuousMultiModalPromptLearner(nn.Module):
    """MaPLe prompt learner with continuous text prompt tokens.

    The text branch receives only the shallow prompt insertion, so those prompt
    token states propagate through all text transformer layers. The visual
    branch keeps MaPLe-style depth-specific projections, but every projection is
    applied to the same trainable text prompt parameter set.
    """

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

        single_layer = nn.Linear(ctx_dim, visual_dim).to(dtype=dtype)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

        print("ContinuousMaPLeMTDA design: continuous text prompts + per-layer visual projections")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context tokens: {n_ctx}")
        print(f"Prompt depth: {self.compound_prompts_depth}")

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        visual_deep_prompts = [
            layer(self.ctx) for layer in self.compound_prompt_projections
        ]
        return (
            prompts,
            self.proj(self.ctx),
            [],
            visual_deep_prompts,
        )


class ContinuousSharedProjPromptLearner(ContinuousMultiModalPromptLearner):
    """Continuous text prompts with one shared deep visual projection."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        maple_cfg = cfg.TRAINER.MAPLE_MTDA
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.conv1.out_channels

        del self.compound_prompt_projections
        self.compound_prompt_projection = nn.Linear(ctx_dim, visual_dim).to(dtype=dtype)

        print(
            "ContinuousSharedProjMaPLeMTDA design: continuous text prompts "
            "+ shared deep visual projection"
        )
        print(f"Shared projection reused for {maple_cfg.PROMPT_DEPTH - 1} deep visual layers")

    def forward_with_ctx(self, ctx_base):
        ctx = ctx_base
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        shared_deep_prompt = self.compound_prompt_projection(ctx_base)
        visual_deep_prompts = [
            shared_deep_prompt for _ in range(self.compound_prompts_depth - 1)
        ]
        return (
            prompts,
            self.proj(ctx_base),
            [],
            visual_deep_prompts,
        )

    def forward(self):
        return self.forward_with_ctx(self.ctx)


class ContinuousSharedProjGapPromptLearner(ContinuousSharedProjPromptLearner):
    """Shared-projection continuous prompt learner with gap-conditioned ctx."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        gap_cfg = cfg.TRAINER.MAPLE_MTDA.GAP_CTX
        ctx_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.conv1.out_channels
        depth = len(clip_model.visual.transformer.resblocks)
        style_layers = [int(layer) for layer in gap_cfg.STYLE_LAYERS]
        if not style_layers:
            raise ValueError("TRAINER.MAPLE_MTDA.GAP_CTX.STYLE_LAYERS cannot be empty")
        for layer in style_layers:
            if layer < 1 or layer > depth:
                raise ValueError(
                    "TRAINER.MAPLE_MTDA.GAP_CTX.STYLE_LAYERS should contain "
                    f"1-based layer ids in [1, {depth}], got {layer}"
                )

        self.gap_style_layers = tuple(layer - 1 for layer in style_layers)
        self.gap_style_eps = float(gap_cfg.STYLE_EPS)
        self.gap_alpha = float(gap_cfg.ALPHA)
        gap_dim = len(style_layers) * visual_dim * 2
        hidden_dim = int(gap_cfg.HIDDEN_DIM)
        self.gap_mlp = nn.Sequential(
            nn.Linear(gap_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_ctx * ctx_dim),
        )

        print("ContinuousSharedProjGapCtxPLMaPLeMTDA design: PL-only gap-conditioned ctx")
        print(f"Gap style layers: {style_layers}")
        print(f"Gap conditioner hidden dim: {hidden_dim}")
        print(f"Gap alpha: {self.gap_alpha}")

    def condition_ctx(self, gap_feature):
        delta = self.gap_mlp(gap_feature.float())
        delta = delta.reshape(self.n_ctx, self.ctx.shape[-1]).to(
            dtype=self.ctx.dtype, device=self.ctx.device
        )
        return self.ctx + self.gap_alpha * delta


class CustomContinuousMaPLeMTDA(CustomMaPLeMTDA):
    prompt_learner_cls = ContinuousMultiModalPromptLearner
    log_prefix = "ContinuousMaPLeMTDA"

    def __init__(self, cfg, classnames, clip_model):
        nn.Module.__init__(self)
        maple_cfg = cfg.TRAINER.MAPLE_MTDA
        self.prompt_learner = self.prompt_learner_cls(
            cfg, classnames, clip_model
        )
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

        print(f"{self.log_prefix} pseudo-label weight: {self.lambda_pl}")
        print(f"{self.log_prefix} pseudo-label final weight: {self.lambda_pl_final}")
        print(f"{self.log_prefix} pseudo-label schedule: {self.pl_schedule}")
        print(f"{self.log_prefix} pseudo-label threshold: {self.pl_threshold}")
        print(f"{self.log_prefix} pseudo-label student threshold: {self.pl_student_threshold}")
        print(
            f"{self.log_prefix} pseudo-label low-conf only: "
            f"{self.pl_use_student_low_conf_mask}"
        )
        print(f"{self.log_prefix} pseudo-label variant: {self.pl_variant}")
        print(
            f"{self.log_prefix} dual-confidence threshold: "
            f"{self.pl_dual_conf_threshold}"
        )
        print(f"{self.log_prefix} soft pseudo-label beta: {self.pl_soft_beta}")
        print(
            f"{self.log_prefix} student-soft weight: "
            f"{self.pl_student_soft_lambda}"
        )
        print(f"{self.log_prefix} weak PL enabled: {self.weak_pl_enabled}")
        print(f"{self.log_prefix} weak PL weight: {self.lambda_weak_pl}")
        print(
            f"{self.log_prefix} weak PL teacher thresholds: "
            f"[{self.weak_pl_teacher_threshold}, {self.weak_pl_teacher_threshold_high})"
        )
        print(f"{self.log_prefix} weak PL student threshold: {self.weak_pl_student_threshold}")
        print(f"{self.log_prefix} weak PL class fraction: {self.weak_pl_fraction}")
        print(f"{self.log_prefix} self-distill enabled: {self.self_distill_enabled}")
        print(f"{self.log_prefix} self-distill weight: {self.lambda_self_distill}")
        print(f"{self.log_prefix} self-distill mode: {self.self_distill_mode}")
        print(f"{self.log_prefix} self-distill temperature: {self.self_distill_temperature}")
        print(
            f"{self.log_prefix} self-distill old confidence band: "
            f"[{self.self_distill_old_conf_low}, {self.self_distill_old_conf_high})"
        )
        print(f"{self.log_prefix} zero-shot prompt template: {zs_template}")


class CustomContinuousSharedProjMaPLeMTDA(CustomContinuousMaPLeMTDA):
    prompt_learner_cls = ContinuousSharedProjPromptLearner
    log_prefix = "ContinuousSharedProjMaPLeMTDA"


class CustomContinuousSharedProjGapCtxPLMaPLeMTDA(CustomContinuousSharedProjMaPLeMTDA):
    prompt_learner_cls = ContinuousSharedProjGapPromptLearner
    log_prefix = "ContinuousSharedProjGapCtxPLMaPLeMTDA"

    def _forward_with_ctx(self, image, ctx):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        (
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = self.prompt_learner.forward_with_ctx(ctx)

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
    def _selected_clean_hidden_states(self, image):
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

        selected = []
        selected_layers = set(self.prompt_learner.gap_style_layers)
        inputs = [x, [], 0]
        for index, block in enumerate(visual.transformer.resblocks):
            inputs = block(inputs)
            if index in selected_layers:
                selected.append(inputs[0].permute(1, 0, 2).detach())
        return selected

    @torch.no_grad()
    def _domain_style_feature(self, image):
        hidden_states = self._selected_clean_hidden_states(image)
        features = []
        for hidden in hidden_states:
            hidden = hidden.float()
            mu = hidden.mean(dim=1)
            std = hidden.std(dim=1, unbiased=False).clamp_min(
                self.prompt_learner.gap_style_eps
            )
            features.append(torch.cat([mu, std], dim=-1))
        style = torch.cat(features, dim=-1).mean(dim=0)
        self._ensure_finite("gap_domain_style", style)
        return style.detach()

    def forward_train(self, image_s, label_s, image_u_dict):
        logits_s = self(image_s)
        loss_ce = F.cross_entropy(logits_s, label_s)
        self._ensure_finite("gapctx_source_ce", loss_ce)

        pl_by_domain = {}
        pl_stats_by_domain = {}
        gap_norm_by_domain = {}
        delta_norm_by_domain = {}
        if self.lambda_pl > 0.0:
            source_style = self._domain_style_feature(image_s)
            for domain_name, image_u in image_u_dict.items():
                target_style = self._domain_style_feature(image_u)
                gap_feature = target_style - source_style
                ctx_t = self.prompt_learner.condition_ctx(gap_feature)
                target_logits = self._forward_with_ctx(image_u, ctx_t)
                reference_logits = self._compute_reference_logits(image_u)
                pl_loss, pl_stats = self._pseudo_label_loss(
                    target_logits, reference_logits
                )
                pl_by_domain[domain_name] = pl_loss
                pl_stats_by_domain[domain_name] = pl_stats
                gap_norm_by_domain[domain_name] = gap_feature.float().norm()
                delta_norm_by_domain[domain_name] = (
                    (ctx_t - self.prompt_learner.ctx).float().norm()
                )
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
            gap_norm = torch.stack(list(gap_norm_by_domain.values())).mean()
            delta_ctx_norm = torch.stack(list(delta_norm_by_domain.values())).mean()
        else:
            loss_pl = loss_ce.new_zeros(())
            pl_coverage = loss_ce.new_zeros(())
            pl_clip_conf = loss_ce.new_zeros(())
            pl_student_conf = loss_ce.new_zeros(())
            clip_student_agreement = loss_ce.new_zeros(())
            gap_norm = loss_ce.new_zeros(())
            delta_ctx_norm = loss_ce.new_zeros(())

        loss_total = loss_ce + self.lambda_pl * loss_pl
        self._ensure_finite("gapctx_loss_total", loss_total)

        if self.debug_print_once and not self._debug_printed:
            print("[ContinuousSharedProjGapCtxPLMaPLeMTDA debug]")
            print("source batch shape:", tuple(image_s.shape))
            for domain_name, image_u in image_u_dict.items():
                print(f"target batch shape [{domain_name}]:", tuple(image_u.shape))
            print("target domains:", list(image_u_dict.keys()))
            print("gap style layers:", self.prompt_learner.gap_style_layers)
            print("lambda_pl:", self.lambda_pl)
            print("loss_ce:", float(loss_ce.detach().item()))
            print("loss_pl:", float(loss_pl.detach().item()))
            print("gap_norm:", float(gap_norm.detach().item()))
            print("delta_ctx_norm:", float(delta_ctx_norm.detach().item()))
            self._debug_printed = True

        outputs = {
            "loss": loss_total,
            "source_ce": loss_ce.detach(),
            "loss_pl": loss_pl.detach(),
            "weighted_loss_pl": (self.lambda_pl * loss_pl).detach(),
            "pl_coverage": pl_coverage.detach(),
            "pl_clip_conf": pl_clip_conf.detach(),
            "pl_student_conf": pl_student_conf.detach(),
            "clip_student_agreement": clip_student_agreement.detach(),
            "gap_norm": gap_norm.detach(),
            "delta_ctx_norm": delta_ctx_norm.detach(),
        }
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
        for domain_name, value in gap_norm_by_domain.items():
            outputs[f"gap_norm_{domain_name}"] = value.detach()
        for domain_name, value in delta_norm_by_domain.items():
            outputs[f"delta_ctx_norm_{domain_name}"] = value.detach()
        return outputs


@TRAINER_REGISTRY.register()
class ContinuousMaPLeMTDA(MaPLeMTDA):
    """Continuous-text MaPLe ablation under the Office-Home SS-MTDA protocol."""

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

        print("Building ContinuousMaPLeMTDA custom CLIP")
        self.model = CustomContinuousMaPLeMTDA(cfg, classnames, clip_model)

        print("Freezing CLIP image/text encoders; updating continuous MaPLe prompt learner only")
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
        self.register_model("ContinuousMaPLeMTDA", self.model, self.optim, self.sched)
        self.scaler = None
        if cfg.TRAINER.MAPLE_MTDA.PREC == "amp":
            from torch.cuda.amp import GradScaler

            self.scaler = GradScaler()
        self._finish_maple_post_build_setup()

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        model_path = osp.join(directory, "ContinuousMaPLeMTDA", model_file)
        if not osp.exists(model_path):
            raise FileNotFoundError(f'Model not found at "{model_path}"')

        checkpoint = load_checkpoint_compat(model_path)
        state_dict = checkpoint["state_dict"]
        for key in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
            state_dict.pop(key, None)

        print(f'Loading weights to ContinuousMaPLeMTDA from "{model_path}"')
        load_state_dict_checked(
            self._models["ContinuousMaPLeMTDA"],
            state_dict,
            allowed_missing=(
                "prompt_learner.token_prefix",
                "prompt_learner.token_suffix",
            ),
            context=f"ContinuousMaPLeMTDA checkpoint {model_path}",
        )


@TRAINER_REGISTRY.register()
class ContinuousSharedProjMaPLeMTDA(ContinuousMaPLeMTDA):
    """Continuous-text MaPLe with one shared deep visual projection."""

    model_name = "ContinuousSharedProjMaPLeMTDA"
    custom_model_cls = CustomContinuousSharedProjMaPLeMTDA

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

        print("Building ContinuousSharedProjMaPLeMTDA custom CLIP")
        self.model = self.custom_model_cls(cfg, classnames, clip_model)

        # Mixed-target baseline mode with zero entropy is a genuine source-only
        # probe. Curriculum runs keep MIX_TARGETS=False and retain target loaders.
        mix_targets = bool(cfg.TRAINER.PROMPT_BASELINE_MTDA.MIX_TARGETS)
        self.uses_target_training = not mix_targets or (
            float(cfg.TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT) > 0.0
        )

        print("Freezing CLIP image/text encoders; updating shared-proj continuous MaPLe prompt learner only")
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
        self.register_model(self.model_name, self.model, self.optim, self.sched)
        self.scaler = None
        if cfg.TRAINER.MAPLE_MTDA.PREC == "amp":
            from torch.cuda.amp import GradScaler

            self.scaler = GradScaler()
        self._finish_maple_post_build_setup()

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        model_path = osp.join(directory, self.model_name, model_file)
        if not osp.exists(model_path):
            raise FileNotFoundError(f'Model not found at "{model_path}"')

        checkpoint = load_checkpoint_compat(model_path)
        state_dict = checkpoint["state_dict"]
        for key in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
            state_dict.pop(key, None)

        print(f'Loading weights to {self.model_name} from "{model_path}"')
        load_state_dict_checked(
            self._models[self.model_name],
            state_dict,
            allowed_missing=(
                "prompt_learner.token_prefix",
                "prompt_learner.token_suffix",
            ),
            context=f"{self.model_name} checkpoint {model_path}",
        )


@TRAINER_REGISTRY.register()
class ContinuousSharedProjGapCtxPLMaPLeMTDA(ContinuousSharedProjMaPLeMTDA):
    """Shared-projection continuous MaPLe with PL-only gap-conditioned ctx."""

    model_name = "ContinuousSharedProjGapCtxPLMaPLeMTDA"
    custom_model_cls = CustomContinuousSharedProjGapCtxPLMaPLeMTDA
