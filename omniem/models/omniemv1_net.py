"""``OmniEMV1Net`` — the concrete net for the ``omniemv1`` model arch.

This is **one case model** registered under ``MODEL_ARCH_REGISTRY["omniemv1"]`` (the
ViT-L encoder + STAdapter z-fusion + UNETR head), NOT the ``omniem`` brand itself
(that is :class:`omniem.models.base.OmniEM`, the stable wrapper). Being a concrete
case, it is intentionally ViT-specific (it calls ``self.vit.forward_adapters`` etc.);
a future non-ViT model would be a *different* net registered as a *different* arch.

Design points:

* The backbone is **INJECTED**, not built from a config path:
  :class:`omniem.models.base.OmniEM` builds the vendored ViT via
  :func:`omniem.encoders.dinov2.build.build` and hands it here.
* This net is **inference-only** — there is no weight-cache, backbone-detach, or
  ``requires_grad`` toggling; the OmniEM wrapper and the split-weight loader own IO.
* z-fusion is the ``kernel3d_z`` 3D-conv path only; there is no separate
  pure-z-fusion (``zconv*``) branch and no image-embedding / integrate-block branch.
* The fixed architecture constants (``feature_size=16``, ``dropout_rate=0.1``,
  ``stadapter_channels=128``, ``upsample_method="resize"``, ``norm_name``,
  ``conv_block``, ``res_block``) are module-level constants below.

Submodule names match the on-disk weight keys — the backbone is ``self.vit`` and the
adapters are ``self.adapters`` — so a ``vit.*`` + ``adapters.*`` ``state_dict`` loads
via plain ``load_state_dict(strict=True)``.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch.nn.functional as nnf
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from torch import nn

from omniem.models.adapter import STAdapter
from omniem.models.upsample import UpBlock

# ---- fixed architecture constants ---------------------------------------------
# Baked into the arch and not exposed as config: changing any of these defines a
# different model that would need its own pretrained weights.
FEATURE_SIZE = 16
DROPOUT_RATE = 0.1
NORM_NAME = "instance"
CONV_BLOCK = True
RES_BLOCK = False
STADAPTER_CHANNELS = 128
UPSAMPLE_METHOD = "resize"
KERNEL_XY = 3            # hardcoded 3 in downsample_kernel_size[0..1]
OMNIEM_PATCH = 16        # internal U-Net patch grid (the ``omniem_patch``)


class OmniEMV1Net(nn.Module):
    """OmniEM model — bare backbone (``self.vit``) + UNETR head.

    Args:
        encoder: An already-built vendored ``DinoVisionTransformer`` backbone
            (no YAML, no weight loading here — the OmniEM wrapper owns that).
        out_channels: Decoder output channels (== :attr:`ModelConfig.out_channels`).
        img_z: ``1`` → 2D model (z-kernel 1, no fusion); ``>1`` → 3D model
            (decoder z-kernel = ``kernel3d_z or 1``).
        kernel3d_z: ``None`` (treated as 1) or a positive ODD int when
            ``img_z>1``. ``img_z==1`` requires ``None``.
        resize4emdino: The resize-to-encoder-grid flag.

    The model returns **pure logits** — there is no in-model
    output nonlinearity. Activation is a property of ``config.task_type`` and
    is applied by :meth:`OmniEM.apply_output` (the model-owned output stage).
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        out_channels: int,
        img_z: int = 16,
        kernel3d_z: int | None = None,
        resize4emdino: bool = False,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.resize4emdino = resize4emdino

        # The attribute name drives the state_dict keys: naming the backbone
        # ``vit`` makes its parameters ``vit.*`` so a vit.* checkpoint loads
        # via plain strict=True.
        self.vit = encoder

        in_channels = int(self.vit.patch_embed.in_chans)

        # 2D vs 3D gating on img_z only.
        if img_z == 1:
            downsample_kernel_size = [KERNEL_XY, KERNEL_XY, 1]
            upsample_kernel_size = [2, 2, 1]
        else:
            # default when kernel3d_z is None is 1 (no z-integration);
            # 3 is the local-z setting (mito-3D head).
            if kernel3d_z is None:
                kernel3d_z = 1
            downsample_kernel_size = [KERNEL_XY, KERNEL_XY, kernel3d_z]
            upsample_kernel_size = [2, 2, 1]

        self.img_z = img_z
        hidden_size = self.vit.num_features
        self.num_layers = len(self.vit.blocks)
        # 4 evenly-spaced inner taps for U-Net skip connections.
        self.sample_layers = np.arange(
            (self.num_layers // 4) - 1,
            self.num_layers,
            self.num_layers // 4,
        )
        self.hidden_size = hidden_size
        self.omniem_patch = OMNIEM_PATCH
        self.feat_size_zconv = img_z
        self.img_size = [self.vit.patch_size] * 2 + [img_z]

        # STAdapter always-on, plural ``self.adapters``.
        self.adapters = nn.ModuleList()
        adapter_ps = dict(
            in_channels=hidden_size,
            adapter_channels=STADAPTER_CHANNELS,
            num_register_tokens=self.vit.num_register_tokens,
            dropout=DROPOUT_RATE,
        )
        for _ in range(self.num_layers):
            self.adapters.append(STAdapter(**adapter_ps))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # --- UNETR encoder-side projections -------------------------------------
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=FEATURE_SIZE,
            kernel_size=downsample_kernel_size,
            stride=1,
            norm_name=NORM_NAME,
            res_block=RES_BLOCK,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=FEATURE_SIZE * 2,
            num_layer=2,
            kernel_size=downsample_kernel_size,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=NORM_NAME,
            conv_block=CONV_BLOCK,
            res_block=RES_BLOCK,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=FEATURE_SIZE * 4,
            num_layer=1,
            kernel_size=downsample_kernel_size,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=NORM_NAME,
            conv_block=CONV_BLOCK,
            res_block=RES_BLOCK,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=FEATURE_SIZE * 8,
            num_layer=0,
            kernel_size=downsample_kernel_size,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=NORM_NAME,
            conv_block=CONV_BLOCK,
            res_block=RES_BLOCK,
        )

        # --- decoders -----------------------------------------------------------
        self.decoder5 = UpBlock(
            in_channels=hidden_size,
            out_channels=FEATURE_SIZE * 8,
            kernel_size=downsample_kernel_size,
            norm_name=NORM_NAME,
            res_block=RES_BLOCK,
            upsample_method=UPSAMPLE_METHOD,
            upsample_kernel_size=upsample_kernel_size,
        )
        self.decoder4 = UpBlock(
            in_channels=FEATURE_SIZE * 8,
            out_channels=FEATURE_SIZE * 4,
            kernel_size=downsample_kernel_size,
            norm_name=NORM_NAME,
            res_block=RES_BLOCK,
            upsample_method=UPSAMPLE_METHOD,
            upsample_kernel_size=upsample_kernel_size,
        )
        self.decoder3 = UpBlock(
            in_channels=FEATURE_SIZE * 4,
            out_channels=FEATURE_SIZE * 2,
            kernel_size=downsample_kernel_size,
            norm_name=NORM_NAME,
            res_block=RES_BLOCK,
            upsample_method=UPSAMPLE_METHOD,
            upsample_kernel_size=upsample_kernel_size,
        )
        self.decoder2 = UpBlock(
            in_channels=FEATURE_SIZE * 2,
            out_channels=FEATURE_SIZE,
            kernel_size=downsample_kernel_size,
            norm_name=NORM_NAME,
            res_block=RES_BLOCK,
            upsample_method=UPSAMPLE_METHOD,
            upsample_kernel_size=upsample_kernel_size,
        )
        self.out = UnetOutBlock(
            spatial_dims=3, in_channels=FEATURE_SIZE, out_channels=out_channels
        )

        # --- inner-tap norms + decoder dropout (always-on) --------------------
        self.norm2 = norm_layer(hidden_size)
        self.norm3 = norm_layer(hidden_size)
        self.norm4 = norm_layer(hidden_size)
        self.drop4 = nn.Dropout3d(DROPOUT_RATE)
        self.drop3 = nn.Dropout3d(DROPOUT_RATE / 2)
        self.drop2 = nn.Dropout3d(DROPOUT_RATE / 4)

        # The model returns PURE LOGITS. The in-model
        # ``output_nonlinear`` is REMOVED; activation is a property of
        # ``config.task_type`` applied by :meth:`OmniEM.apply_output` (the
        # output stage), never inside the model forward.

        # Derive the encoder's state_dict prefix from the child module that holds
        # it, BY IDENTITY — so backbone-vs-head partitioning (split save/load,
        # freezing) follows wherever the ``encoder`` actually lives, with no
        # hard-coded name anywhere downstream.
        self._encoder_attr = next(
            name for name, child in self.named_children() if child is encoder
        )

    @property
    def encoder_prefix(self) -> str:
        """The encoder backbone's ``state_dict`` key prefix.

        **Derived** from the child module that holds the encoder (by object
        identity at construction), never hard-coded — so the backbone-vs-head
        partition adapts to whatever attribute the encoder is stored under. Keys
        under ``f"{encoder_prefix}."`` are the backbone; the rest are the head
        (decoder + adapters + ``out``).
        """
        return self._encoder_attr

    # ---- internal helpers --------------------------------------------------

    def proj_feat(self, x, hidden_size, feat_size):
        """Un-flatten ViT patch tokens to a 3D feature volume.

        ``[B*Z, N_patches, H] -> [B, H, X//p, Y//p, Z]``
        """
        x = x.reshape(
            int(x.size(0) / feat_size[2]),
            feat_size[2], feat_size[0], feat_size[1],
            hidden_size,
        )
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        return x

    # ---- forward -----------------------------------------------------------

    def forward(self, x_in):
        """Forward pass.

        Accepts a 4D ``[B, C, X, Y]`` (auto Z=1) or 5D ``[B, C, X, Y, Z]`` input.
        ``C==1`` is repeated to the ViT's ``in_chans`` (= 3) to promote grayscale
        to the encoder's channel count.
        """
        # input preparation ------------------------------------------------------
        L = len(x_in.shape)
        if L == 4:
            B, C, X, Y = tuple(x_in.shape)
            Z = 1
            x_in = x_in.reshape((B, C, X, Y, Z))
        elif L == 5:
            B, C, X, Y, Z = tuple(x_in.shape)
        else:
            raise ValueError(
                f"OmniEMV1Net.forward expects 4D or 5D input; got {L}D"
            )
        # XY must be divisible by ViT patch size
        assert X % self.vit.patch_size == 0
        assert Y % self.vit.patch_size == 0
        if C == 1:
            x_in = x_in.repeat(1, 3, 1, 1, 1)
            C = 3
        assert Z == self.img_size[2]

        # forward ----------------------------------------------------------------
        if self.resize4emdino:
            data_feat_size_zconv = (
                X // self.omniem_patch,
                Y // self.omniem_patch,
                self.feat_size_zconv,
            )
        else:
            data_feat_size_zconv = (
                X // self.vit.patch_size,
                Y // self.vit.patch_size,
                self.img_size[2],
            )
            assert X % self.vit.patch_size == 0
            assert Y % self.vit.patch_size == 0

        enc1 = self.encoder1(x_in)

        x4emdino = x_in.permute(0, 4, 1, 2, 3).reshape(B * Z, C, X, Y)
        if self.resize4emdino:
            if self.vit.patch_size != self.omniem_patch:
                resize_n = X * self.vit.patch_size / self.omniem_patch
                assert int(resize_n) == resize_n
                resize_n = int(resize_n)
                x4emdino = nnf.interpolate(
                    x4emdino, size=(resize_n, resize_n),
                    mode="bicubic", align_corners=False,
                )

        # STAdapter is always-on (pinned) — drives forward_adapters.
        x = self.vit.forward_adapters(x4emdino, self.adapters, T=Z)
        hidden_states_out = x["inner_features"]

        # Inner taps → norm → proj → encoderN.
        x2 = hidden_states_out[self.sample_layers[0]][:, self.vit.num_register_tokens + 1:]
        x2 = self.norm2(x2)
        x2 = self.proj_feat(x2, self.hidden_size, data_feat_size_zconv)
        enc2 = self.encoder2(x2)

        x3 = hidden_states_out[self.sample_layers[1]][:, self.vit.num_register_tokens + 1:]
        x3 = self.norm3(x3)
        x3 = self.proj_feat(x3, self.hidden_size, data_feat_size_zconv)
        enc3 = self.encoder3(x3)

        x4 = hidden_states_out[self.sample_layers[2]][:, self.vit.num_register_tokens + 1:]
        x4 = self.norm4(x4)
        x4 = self.proj_feat(x4, self.hidden_size, data_feat_size_zconv)
        enc4 = self.encoder4(x4)

        # Deepest decoder level uses the final ViT patch tokens.
        x_out = x["x_norm_patchtokens"]
        x_out = self.proj_feat(x_out, self.hidden_size, data_feat_size_zconv)

        dec4 = x_out
        dec4 = self.drop4(dec4)
        dec3 = self.decoder5(dec4, enc4)
        dec3 = self.drop3(dec3)
        dec2 = self.decoder4(dec3, enc3)
        dec2 = self.drop2(dec2)
        dec1 = self.decoder3(dec2, enc2)

        # One unconditional trilinear interp, plus a second identical one when
        # resize4emdino is False. Both target the same shape (a half-resolution
        # XY grid at full Z); the second pass deepens the upsample for that mode.
        dec1 = nnf.interpolate(
            dec1,
            size=(int(enc1.shape[2] / 2), int(enc1.shape[3] / 2), enc1.shape[4]),
            mode="trilinear", align_corners=False,
        )
        if not self.resize4emdino:
            dec1 = nnf.interpolate(
                dec1,
                size=(int(enc1.shape[2] / 2), int(enc1.shape[3] / 2), enc1.shape[4]),
                mode="trilinear", align_corners=False,
            )

        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        # Pure logits — no in-model activation.
        # Activation moved to OmniEM.apply_output (model-owned, task_type-gated).
        return logits


__all__ = ["OmniEMV1Net"]
