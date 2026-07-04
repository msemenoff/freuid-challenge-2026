"""Global vision backbones: DINOv2-large and ConvNeXt-V2-large.

Both expose a forward() method returning a flat feature vector.
Supports frozen (feature extraction) and unfrozen (fine-tune) modes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------------
# ConvNeXt-V2 via timm
# ---------------------------------------------------------------------------

class ConvNeXtV2Backbone(nn.Module):
    """ConvNeXt-V2-large global feature extractor.

    Output: (batch, embed_dim)  where embed_dim = 1536 for -large.
    """

    def __init__(
        self,
        model_name: str = "convnextv2_large",
        pretrained: bool = True,
        frozen: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove classifier head
            global_pool="avg",
        )
        self.embed_dim: int = self.model.num_features
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        if frozen:
            self.freeze()

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze(self, stages: int | None = None) -> None:
        """Unfreeze last `stages` stages (None = all)."""
        if stages is None:
            for p in self.model.parameters():
                p.requires_grad_(True)
        else:
            # timm ConvNeXt has stages 0-3; unfreeze from the end
            n_stages = len(self.model.stages)
            start = max(0, n_stages - stages)
            for i in range(start, n_stages):
                for p in self.model.stages[i].parameters():
                    p.requires_grad_(True)
            # always unfreeze head norm
            if hasattr(self.model, "head"):
                for p in self.model.head.parameters():
                    p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)          # (B, embed_dim)
        return self.dropout(feats)


# ---------------------------------------------------------------------------
# DINOv2 via HuggingFace transformers
# ---------------------------------------------------------------------------

class DINOv2Backbone(nn.Module):
    """DINOv2-large backbone returning CLS token and/or patch statistics.

    Output shape depends on flags:
        - use_cls          → (B, 1024)
        - use_patch_mean   → (B, 1024)
        - use_patch_stats  → (B, 1024*2)  [max + std per dim]
    All enabled features are concatenated along dim=1.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        frozen: bool = True,
        use_cls: bool = True,
        use_patch_mean: bool = True,
        use_patch_stats: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel
        self.dino = AutoModel.from_pretrained(model_name)
        self.hidden_size: int = self.dino.config.hidden_size  # 1024 for -large

        self.use_cls = use_cls
        self.use_patch_mean = use_patch_mean
        self.use_patch_stats = use_patch_stats

        # Compute output dim
        self.embed_dim = 0
        if use_cls:
            self.embed_dim += self.hidden_size
        if use_patch_mean:
            self.embed_dim += self.hidden_size
        if use_patch_stats:
            self.embed_dim += self.hidden_size * 2  # max + std

        if frozen:
            self.freeze()

    def freeze(self) -> None:
        for p in self.dino.parameters():
            p.requires_grad_(False)

    def unfreeze_last_n_blocks(self, n: int = 4) -> None:
        for block in self.dino.encoder.layer[-n:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for p in self.dino.layernorm.parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.dino(pixel_values=x, output_hidden_states=False)
        # last_hidden_state: (B, 1 + n_patches, hidden_size)
        hidden = outputs.last_hidden_state

        cls_token = hidden[:, 0, :]           # (B, H)
        patch_tokens = hidden[:, 1:, :]       # (B, N, H)

        parts = []
        if self.use_cls:
            parts.append(cls_token)
        if self.use_patch_mean:
            parts.append(patch_tokens.mean(dim=1))
        if self.use_patch_stats:
            parts.append(patch_tokens.max(dim=1).values)
            parts.append(patch_tokens.std(dim=1))

        return torch.cat(parts, dim=1)        # (B, embed_dim)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_global_backbone(cfg) -> nn.Module:
    return ConvNeXtV2Backbone(
        model_name=cfg.global_backbone.name,
        pretrained=cfg.global_backbone.pretrained,
        frozen=cfg.global_backbone.frozen,
        dropout=cfg.global_backbone.dropout,
    )


def build_dinov2(cfg) -> nn.Module:
    return DINOv2Backbone(
        model_name=cfg.dinov2.model_name,
        frozen=cfg.dinov2.frozen,
        use_cls=cfg.dinov2.use_cls,
        use_patch_mean=cfg.dinov2.use_patch_mean,
        use_patch_stats=cfg.dinov2.use_patch_stats,
    )
