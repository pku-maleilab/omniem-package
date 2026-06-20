"""Spatio-temporal adapter (``STAdapter``) — the head-owned z-fusion module.

The adapter is the head-owned z-fusion module. It is injected **after every ViT block**
by the backbone's ``forward_adapters(x, adapters, T=Z)`` loop (see
``omniem/encoders/dinov2/backbone.py``): for each block ``i`` the block output ``x`` is
passed through ``adapters[i]`` before feeding block ``i+1``.

Mechanism (one adapter call):
* ``x`` is ``[B*T, L, C]`` where ``T = Z`` (z-slices folded into the batch), ``L`` = 1 cls
  + ``num_register_tokens`` + ``H*W`` patch tokens, ``C`` = embed dim.
* strip cls + register tokens, bottleneck down (``fc1``: C -> adapter_channels), reshape the
  flat batch back to ``[B, adapter_channels, T, H, W]``, apply a **depthwise Conv3d with
  z-kernel (3,1,1)** (fuses 3 consecutive z-slices; same-padded so shape is preserved),
  reshape back, bottleneck up (``fc2``), and add as a **residual** at the patch
  positions (rebuilt out-of-place — see the NOTE below).
* At ``T == 1`` (a 2D model, ``img_z==1``) the (3,1,1) conv reduces to its center-tap weight
  — a per-channel 1x1 op, degenerate but nonzero; the same code path handles 2D and 3D.

Conv weights are zero-initialised, so an untrained adapter is the identity (the residual is
zero); the trained weights ride in the head checkpoint under ``adapters.*``
(the on-disk attribute name; no key renaming).

NOTE: this code assumes ``num_register_tokens == 0`` for the reshape
``view(BT, L - 1, Ca)`` to be exact — which holds for the EM-DINO heads.

The residual is applied **out-of-place** (rebuilt via ``torch.cat``) rather than as an
in-place ``x_id[..., patch] += x``: an in-place add mutates the adapter's input after
``fc1`` has saved a view of it for backward, tripping autograd's version check once the
adapter is trainable (``prepare_train`` leaves the adapter in the trainable head). The
out-of-place rebuild is numerically identical (the conv is zero-init, so an untrained
adapter is exactly the identity) and is backprop-safe.
"""

import math

import torch
import torch.nn as nn


class STAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels,
                 kernel_size=(3, 1, 1), num_register_tokens=0, dropout=0):
        super().__init__()
        self.num_register_tokens = num_register_tokens
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,  # depthwise
        )
        self.dropout = nn.Dropout3d(dropout)
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        # Zero-init the conv (residual adapter = identity until trained); zero the
        # bottleneck biases so an untrained adapter contributes exactly zero.
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, T):
        # x: [B*T, L, C]; T = number of z-slices folded into the batch.
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        # spatial patch grid side (cls + register tokens excluded)
        H = W = round(math.sqrt(L - 1 - self.num_register_tokens))
        assert L - 1 - self.num_register_tokens == H * W
        x_id = x
        # drop cls (pos 0) + register tokens, keep patch tokens
        x = x[:, 1 + self.num_register_tokens:, :]
        x = self.fc1(x)                                              # [BT, H*W, Ca]
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()  # [B, Ca, T, H, W]
        x = self.conv(x)                                            # z-fuse (same-padded)
        x = self.dropout(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x)                                             # [BT, H*W, C]
        # Out-of-place residual at the patch positions. An in-place ``+=`` on x_id
        # mutates the adapter's input tensor after ``fc1`` saved a view of it for
        # backward, so when the adapter trains (it is a head-owned, trainable
        # module) autograd's version check fails ("a variable needed for gradient
        # computation has been modified by an inplace operation"). Rebuilding the
        # sequence out-of-place — cls + register tokens passed through unchanged,
        # patch tokens carrying the residual — is numerically identical to the
        # in-place add and is backprop-safe.
        prefix = x_id[:, : 1 + self.num_register_tokens, :]          # cls + register tokens
        patch = x_id[:, 1 + self.num_register_tokens:, :] + x        # residual on patch tokens
        return torch.cat([prefix, patch], dim=1)
