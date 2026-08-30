# Ported from BaRISTA: https://github.com/ShanechiLab/BaRISTA
# USC research/non-profit license: LICENSES/BaRISTA-LICENSE.md

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
from typing import List

from imindbench.models.barista_components.atlas import (
    destrieux_label_to_region_enum,
    DestrieuxAseg,
)
from imindbench.models import register_model
from imindbench.models.torch_base_model import TorchBaseModel
from imindbench.models.barista_components.tokenizer import Tokenizer
from imindbench.models.barista_components.transformer import Transformer
from imindbench.utils.logging_utils import log


class BaristaNetwork(nn.Module):
    def __init__(
        self,
        model_config: DictConfig,
        input_shape=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.tokenizer = self._build_tokenizer(
            model_config.tokenizer,
            input_shape=input_shape,
        )

        self.backbone = Transformer(
            **model_config.backbone,
        )

        self.d_hidden = model_config.backbone.d_hidden

        self.head = None

    def _build_tokenizer(self, tokenizer_cfg, input_shape=None):
        tokenizer_kind = str(
            tokenizer_cfg.get("kind", tokenizer_cfg.get("type", "region"))
        ).lower()
        if tokenizer_kind != "region":
            raise ValueError(
                f"Unsupported model.tokenizer.kind={tokenizer_kind!r}; "
                "only 'region' is available in this release."
            )
        return Tokenizer(config=tokenizer_cfg)

    def create_downstream_head(self, n_chans, output_dim):
        self.channel_weights = nn.Linear(
            n_chans * self.tokenizer.num_subsegments,
            1,
            bias=False,
        )
        self.binary_classifier = nn.Linear(self.d_hidden, output_dim)

    def get_latent_embeddings(
        self, x: torch.Tensor, subject_sessions: List, region_enum_ids: torch.Tensor
    ):
        #  Get tokens
        tokenized_x = self.tokenizer(
            x, subject_sessions, region_enum_ids, output_as_list=False
        )

        # Pass through transformer
        latents = self.backbone(
            x=tokenized_x.tokens,
            seq_lens=tokenized_x.seq_lens,
            position_ids=tokenized_x.position_ids,
        )

        return latents

    def forward(self, x: torch.Tensor, *args, **model_kwargs):
        recording_ids = model_kwargs["recording_ids"]
        region_ids = model_kwargs["region_ids"]
        latents = self.get_latent_embeddings(
            x, subject_sessions=recording_ids, region_enum_ids=region_ids
        )

        # Pass through Task head
        batch_size = x.shape[0]
        latents_reshaped = latents.reshape(batch_size, -1, latents.shape[-1])
        x = self.channel_weights(latents_reshaped.permute(0, 2, 1)).squeeze(dim=-1)
        x = self.binary_classifier(x)

        return x

    def get_task_params(self):
        return [
            *self.channel_weights.named_parameters(),
            *self.binary_classifier.named_parameters(),
        ]

    def get_upstream_params(self):
        return [*self.tokenizer.named_parameters(), *self.backbone.named_parameters()]


@register_model("barista")
class Barista(TorchBaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        self.accepts_coords = True
        self.upstream_ckpt = cfg.get("upstream_ckpt", None)
        self.random_init = cfg.get("random_init", False)
        if self.upstream_ckpt is None and not self.random_init:
            raise ValueError(
                "upstream_ckpt must be provided unless model.random_init=true"
            )

        self.frozen_upstream = cfg.get("frozen_upstream", False)
        self.freeze_tokenizer_temporal = cfg.get("freeze_tokenizer_temporal", True)
        self.upstream_lr = cfg.get("upstream_lr", 1e-4)
        self.head_lr = cfg.get("head_lr", 1e-3)

        self.upstream = None
        self.classification_head = None

        self._region_enum_cache = {}
        self._network_cls = BaristaNetwork

    def build_model(self, input_shape, n_classes, device=None):
        """Build the model with given input shape and number of classes."""
        self._resolve_device(device)

        self.model = self._create_network(input_shape, n_classes)
        self.model = self.model.to(self.device)

        self.classes_ = np.arange(n_classes)
        return self.model

    def _create_network(self, input_shape, n_classes):
        """Create the neural network architecture."""
        model = self._network_cls(self.cfg, input_shape=input_shape)
        if self.random_init:
            log("[Barista] Using RANDOMLY INITIALIZED weights", priority=1)
        else:
            model = self._load_upstream_from_checkpoint(model, self.upstream_ckpt)

        if self.frozen_upstream:
            for p in model.tokenizer.parameters():
                p.requires_grad = False
            for p in model.backbone.parameters():
                p.requires_grad = False
            log("[Barista] Upstream (tokenizer + backbone) fully frozen", priority=1)
        elif self.freeze_tokenizer_temporal:
            num_frozen, num_trainable = 0, 0
            for name, p in model.tokenizer.named_parameters():
                if name.startswith("spatial_encoder"):
                    num_trainable += p.numel()
                    continue
                p.requires_grad = False
                num_frozen += p.numel()
            log(
                "[Barista] Tokenizer CNN + temporal pooler frozen "
                f"({num_frozen} params); spatial_encoder trainable "
                f"({num_trainable} params); transformer backbone trainable.",
                priority=1,
            )
        else:
            num_trainable = sum(
                p.numel()
                for p in (*model.tokenizer.parameters(), *model.backbone.parameters())
                if p.requires_grad
            )
            log(
                "[Barista] Tokenizer + transformer backbone fully trainable "
                f"({num_trainable} params).",
                priority=1,
            )

        model.create_downstream_head(n_chans=input_shape[0], output_dim=n_classes)

        return model

    def get_parameter_groups(self):
        if self.model is None:
            raise RuntimeError(
                "get_parameter_groups() called before build_model(); "
                "self.model is None."
            )

        head_modules = [self.model.binary_classifier]
        if hasattr(self.model, "channel_weights"):
            head_modules.append(self.model.channel_weights)
        head_param_ids = set()
        head_params = []
        for m in head_modules:
            for p in m.parameters():
                if not p.requires_grad:
                    continue
                head_params.append(p)
                head_param_ids.add(id(p))

        upstream_params = [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in head_param_ids
        ]

        param_groups = []
        if upstream_params:
            param_groups.append({"params": upstream_params, "lr": self.upstream_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr})

        log(
            f"[Barista] Parameter groups: upstream_lr={self.upstream_lr}, "
            f"head_lr={self.head_lr}",
            priority=1,
        )
        return param_groups

    def prepare_batch(self, batch, **kwargs):

        batch["model_kwargs"] = {
            "region_ids": self._get_region_enum_ids(
                all_brain_areas=batch["brain_areas"],
                recording_ids=batch["recording_ids"],
            ),
            "recording_ids": batch["recording_ids"],
        }

        return super().prepare_batch(batch, **kwargs)

    def _get_region_enum_ids(
        self, recording_ids: List[str], all_brain_areas: List[List[str]]
    ):
        if not self.cfg.tokenizer.add_spatial_encoding:
            return torch.zeros(len(recording_ids))

        if all_brain_areas is None:
            # Upstream returned an all-UNKNOWN tensor here, which meant an unset
            # dataset.brain_area_key silently collapsed the spatial encoding --
            # the same failure the Destrieux check below guards against, but
            # reached before any label exists to inspect. Several dataset
            # configs (e.g. berezutskayapippi2022) default the key to null.
            raise ValueError(
                "Barista needs per-channel brain regions to build its spatial "
                "encoding, but the batch carried no brain_areas. Set "
                "dataset.brain_area_key to a Destrieux-labelled key -- "
                "'localization_Destrieux' for neuroprobev2, 'label_destrieux' "
                "for pippi and BYD-destrieux, or 'location' for BYD -- or set "
                "model.tokenizer.add_spatial_encoding=false to run without "
                "spatial encoding on purpose."
            )

        all_enum_ids = []
        for recording_id, brain_areas in zip(recording_ids, all_brain_areas):
            if recording_id in self._region_enum_cache:
                enum_ids = self._region_enum_cache[recording_id]
            else:
                enum_ids = [destrieux_label_to_region_enum(b) for b in brain_areas]
                self._check_region_labels_are_destrieux(
                    recording_id=recording_id,
                    brain_areas=brain_areas,
                    enum_ids=enum_ids,
                )
                self._region_enum_cache[recording_id] = enum_ids

            all_enum_ids.append(enum_ids)

        all_enum_ids = torch.tensor(all_enum_ids)
        return all_enum_ids

    def _check_region_labels_are_destrieux(
        self, *, recording_id: str, brain_areas: List[str], enum_ids: List[int]
    ):
        """Fail loudly when channel labels are not from the Destrieux atlas.

        The spatial encoder is keyed on Destrieux region names. Labels from
        another parcellation (e.g. Desikan-Killiany, whose names look like
        `ctx-lh-superiortemporal` rather than `ctx_lh_G_temp_sup-Lateral`)
        fall through to UNKNOWN, so the model trains without complaint on a
        collapsed spatial encoding and reports plausible but wrong numbers.

        The two cases separate cleanly rather than sitting on a threshold: on
        neuroprobev2 sub1_sess1, Destrieux labels resolve 120/120 channels
        while Desikan-Killiany resolves 7/120 -- only the subcortical names,
        which happen to be spelled the same in both atlases. So "most channels
        failed to resolve" is a wiring mistake, not a localization gap.
        """
        unknown = DestrieuxAseg.UNKNOWN.value
        n_total = len(enum_ids)
        if n_total == 0:
            return

        n_resolved = sum(1 for e in enum_ids if e != unknown)
        fraction = n_resolved / n_total
        min_fraction = float(self.cfg.get("min_resolved_region_fraction", 0.5))

        if fraction >= min_fraction:
            log(
                f"[Barista] {recording_id}: {n_resolved}/{n_total} channels "
                "resolved to a Destrieux region.",
                priority=2,
            )
            return

        examples = sorted({str(b) for b in brain_areas})[:5]
        raise ValueError(
            f"Barista resolved only {n_resolved}/{n_total} channel labels "
            f"({fraction:.1%}) for recording '{recording_id}' onto the "
            "Destrieux atlas its spatial encoder was pretrained on, below the "
            f"required {min_fraction:.0%}. Example labels: {examples}. This "
            "usually means dataset.brain_area_key points at the wrong "
            "parcellation -- use 'localization_Destrieux' for neuroprobev2, "
            "'label_destrieux' for pippi and BYD-destrieux, or 'location' for "
            "BYD. Set model.tokenizer.add_spatial_encoding=false to run "
            "without spatial encoding on purpose, or lower "
            "model.min_resolved_region_fraction if the gap is genuine."
        )

    def _load_upstream_from_checkpoint(self, model, checkpoint_path):
        """Load pretrained weights, excluding masked_recon and multi_head_fc layers."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )

        pretrained_spatial_encoder_weights = checkpoint[
            "tokenizer.spatial_encoder.subcomponent_embeddings.0.weight"
        ]
        model_spatial_encoder_weights = (
            model.tokenizer.spatial_encoder.subcomponent_embeddings["0"].weight
        )
        # Extend checkpoint embeddings for atlas regions added by this port.
        if (
            pretrained_spatial_encoder_weights.shape[0]
            != model_spatial_encoder_weights.shape[0]
        ):
            new_embeddings = torch.empty(
                model_spatial_encoder_weights.shape[0]
                - pretrained_spatial_encoder_weights.shape[0],
                model_spatial_encoder_weights.shape[-1],
                device=self.device,
            )
            nn.init.normal_(new_embeddings)
            new_data = torch.cat(
                (pretrained_spatial_encoder_weights.data, new_embeddings)
            )
            checkpoint["tokenizer.spatial_encoder.subcomponent_embeddings.0.weight"] = (
                new_data
            )

        model.load_state_dict(checkpoint)
        log("[Barista] Loaded pretrained checkpoint weights", priority=1)
        print(f"Pretrained weights loaded from {checkpoint_path}")
        return model
