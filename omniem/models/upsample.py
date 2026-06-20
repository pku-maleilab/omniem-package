"""Decoder upsampling blocks for the UNETR head.

``UpBlock`` is the UNETR decoder stage: upsample (transposed conv / sub-pixel / resize —
all XY-only, Z preserved), concat the skip connection, then a MONAI ``UnetBasicBlock`` /
``UnetResBlock``. ``ESPCN3D_XY`` and ``ResizeConv3D_XY`` are pure-torch XY-only upsamplers.

The MONAI ``dynunet_block`` primitives (``get_conv_layer``, ``UnetBasicBlock``,
``UnetResBlock``) produce identical numerics across the supported monai range
(1.2.0–1.5.2).
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from torch import nn


class ESPCN3D_XY(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
    ):
        super().__init__()
        channels = out_channels * (upscale_factor ** 2)
        hidden_channels = channels // 2
        # Only upscale in X/Y, not Z
        out_channels = int(out_channels * (upscale_factor ** 2))

        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=(5, 5, 3), stride=(1, 1, 1), padding=(2, 2, 1)),
            nn.Tanh(),
            nn.Conv3d(channels, hidden_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Tanh(),
        )

        # Sub-pixel convolution (upsampling only in XY)
        self.sub_pixel = nn.Conv3d(hidden_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.upscale_factor = upscale_factor

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight.data,
                                0.0,
                                math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)

    def pixel_shuffle_xy(self, x: torch.Tensor) -> torch.Tensor:
        """Custom pixel shuffle along X and Y only."""
        b, c, h, w, d = x.size()
        r = self.upscale_factor
        out_c = c // (r ** 2)
        # Reshape: separate subpixel groups
        x = x.view(b, out_c, r, r, h, w, d)
        # Rearrange: interleave r×r patches into X/Y
        x = x.permute(0, 1, 5, 2, 6, 3, 4).contiguous()  # [B, out_c, h, r, w, r, d]
        x = x.view(b, out_c, h * r, w * r, d)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_maps(x)
        x = self.sub_pixel(x)
        x = self.pixel_shuffle_xy(x)
        return x


class ResizeConv3D_XY(nn.Module):
    """
    Resize+Conv that upsamples only in X/Y while keeping Z unchanged.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        upscale_factor (int): Upscaling factor for X/Y dimensions.
    """
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.scale = upscale_factor

        # Expand features (before interpolation)
        self.expand = nn.Conv3d(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1
        )

        # Fuse after interpolation (reduce channels & smooth)
        self.fuse = nn.Conv3d(
            out_channels * (upscale_factor ** 2),
            out_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x):
        # x: [B, C, X, Y, Z]
        x = self.expand(x)  # [B, out_ch * s^2, X, Y, Z]

        # Upsample only in X and Y, keep Z
        x = F.interpolate(
            x,
            scale_factor=(self.scale, self.scale, 1),
            mode="trilinear",
            align_corners=False
        )

        # Fuse redundant channels
        x = self.fuse(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
        upsample_method: str = "conv",
        upsample_kernel_size: Sequence[int] | int = None,
    ) -> None:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
                if it is None, use suppixel upsample instead of conv_layer

        """

        super().__init__()
        spatial_dims = 3
        if upsample_method == "conv":
            assert upsample_kernel_size is not None
            upsample_stride = upsample_kernel_size
            self.transp = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_stride,
                conv_only=True,
                is_transposed=True,
            )
        elif upsample_method == "subpixel":
            self.transp = ESPCN3D_XY(
                in_channels=in_channels,
                out_channels=out_channels,
                upscale_factor=2,
            )
        elif upsample_method == "resize":
            self.transp = ResizeConv3D_XY(
                in_channels,
                out_channels,
                upscale_factor=2,
            )
        else:
            raise Exception()
        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
