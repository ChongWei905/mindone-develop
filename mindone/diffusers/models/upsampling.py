# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple

import mindspore as ms
from mindspore import nn, ops

from .layers_compat import conv_transpose2d, pad, upsample_nearest3d_free_interpolate
from .normalization import LayerNorm, RMSNorm


class Upsample1D(nn.Cell):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.Conv1dTranspose(channels, self.out_channels, 4, 2, pad_mode="pad", padding=1, has_bias=True)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, pad_mode="pad", padding=1, has_bias=True)

    def construct(self, inputs: ms.Tensor) -> ms.Tensor:
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        # Only 'area' mode supports argument 'scale_factor', use 'recompute_scale_factor=True'
        # refer to: https://www.mindspore.cn/docs/zh-CN/r2.2/note/api_mapping/pytorch_diff/interpolate.html
        outputs = ops.interpolate(inputs, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class Upsample2D(nn.Cell):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if norm_type == "ln_norm":
            self.norm = LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.Conv2dTranspose(
                channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=2,
                pad_mode="pad",
                padding=padding,
                has_bias=bias,
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                pad_mode="pad",
                padding=padding,
                has_bias=bias,
            )

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def construct(self, hidden_states: ms.Tensor, output_size: Optional[int] = None) -> ms.Tensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == ms.bfloat16:
            hidden_states = hidden_states.to(ms.float32)

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            if output_size is None:
                _, _, h, w = hidden_states.shape
                hidden_states = ops.interpolate(hidden_states, size=(h * 2, w * 2), mode="nearest")
            else:
                hidden_states = ops.interpolate(hidden_states, size=output_size, mode="nearest")

        # Cast back to original dtype
        if dtype == ms.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class FirUpsample2D(nn.Cell):
    """A 2D FIR upsampling layer with an optional convolution.

    Parameters:
        channels (`int`, optional):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        fir_kernel (`tuple`, default `(1, 3, 3, 1)`):
            kernel for the FIR filter.
    """

    def __init__(
        self,
        channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        fir_kernel: Tuple[int, int, int, int] = (1, 3, 3, 1),
    ):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = nn.Conv2d(
                channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
            )
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def _upsample_2d(
        self,
        hidden_states: ms.Tensor,
        weight: Optional[ms.Tensor] = None,
        kernel: Optional[ms.Tensor] = None,
        factor: int = 2,
        gain: float = 1,
    ) -> ms.Tensor:
        """Fused `upsample_2d()` followed by `Conv2d()`.

        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states (`ms.Tensor`):
                Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight (`ms.Tensor`, *optional*):
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel (`ms.Tensor`, *optional*):
                FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
                corresponds to nearest-neighbor upsampling.
            factor (`int`, *optional*): Integer upsampling factor (default: 2).
            gain (`float`, *optional*): Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output (`ms.Tensor`):
                Tensor of the shape `[N, C, H * factor, W * factor]` or `[N, H * factor, W * factor, C]`, and same
                datatype as `hidden_states`.
        """

        assert isinstance(factor, int) and factor >= 1

        # Setup filter kernel.
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = ms.Tensor(kernel, dtype=ms.float32)
        if kernel.ndim == 1:
            kernel = ops.outer(kernel, kernel)
        kernel /= ops.sum(kernel)

        kernel = kernel * (gain * (factor**2))

        if self.use_conv:
            convH = weight.shape[2]
            convW = weight.shape[3]
            inC = weight.shape[1]

            pad_value = (kernel.shape[0] - factor) - (convW - 1)

            stride = (factor, factor)
            # Determine data dimensions.
            output_shape = (
                (hidden_states.shape[2] - 1) * factor + convH,
                (hidden_states.shape[3] - 1) * factor + convW,
            )
            output_padding = (
                output_shape[0] - (hidden_states.shape[2] - 1) * stride[0] - convH,
                output_shape[1] - (hidden_states.shape[3] - 1) * stride[1] - convW,
            )
            assert output_padding[0] >= 0 and output_padding[1] >= 0
            num_groups = hidden_states.shape[1] // inC

            # Transpose weights.
            weight = ops.reshape(weight, (num_groups, -1, inC, convH, convW))
            weight = ops.flip(weight, dims=[3, 4]).permute(0, 2, 1, 3, 4)
            weight = ops.reshape(weight, (num_groups * inC, -1, convH, convW))

            inverse_conv = conv_transpose2d(
                hidden_states,
                weight,
                stride=stride,
                output_padding=output_padding,
                padding=0,
            )

            output = upfirdn2d_native(
                inverse_conv,
                kernel,
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2 + 1),
            )
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                kernel,
                up=factor,
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
            )

        return output

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        if self.use_conv:
            height = self._upsample_2d(hidden_states, self.Conv2d_0.weight, kernel=self.fir_kernel)
            height = height + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            height = self._upsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)

        return height


class KUpsample2D(nn.Cell):
    r"""A 2D K-upsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = ms.Tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]]) * 2
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.kernel = kernel_1d.T @ kernel_1d

    def construct(self, inputs: ms.Tensor) -> ms.Tensor:
        inputs = pad(inputs, ((self.pad + 1) // 2,) * 4, self.pad_mode)
        weight = inputs.new_zeros(
            [
                inputs.shape[1],
                inputs.shape[1],
                self.kernel.shape[0],
                self.kernel.shape[1],
            ]
        )
        indices = ops.arange(inputs.shape[1])
        kernel = self.kernel.to(weight.dtype)[None, :].broadcast_to((inputs.shape[1], -1, -1))
        weight[indices, indices] = kernel
        return conv_transpose2d(inputs, weight, stride=2, padding=self.pad * 2 + 1)


class CogVideoXUpsample3D(nn.Cell):
    r"""
    A 3D Upsample layer using in CogVideoX by Tsinghua University & ZhipuAI # Todo: Wait for paper relase.

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `1`):
            Stride of the convolution.
        padding (`int`, defaults to `1`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pad_mode="pad",
            has_bias=True,
        )
        self.compress_time = compress_time

    def construct(self, inputs: ms.Tensor) -> ms.Tensor:
        if self.compress_time:
            if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
                # split first frame
                x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]

                x_first = upsample_nearest3d_free_interpolate(x_first, scale_factor=2.0)
                x_rest = upsample_nearest3d_free_interpolate(x_rest, scale_factor=2.0)
                x_first = x_first[:, :, None, :, :]
                inputs = ops.cat([x_first, x_rest], axis=2)
            elif inputs.shape[2] > 1:
                inputs = upsample_nearest3d_free_interpolate(inputs, scale_factor=2.0)
            else:
                if inputs.shape[2] == 1:
                    inputs = inputs.squeeze(2)
                inputs = upsample_nearest3d_free_interpolate(inputs, scale_factor=2.0)
                inputs = inputs[:, :, None, :, :]
        else:
            # only interpolate 2D
            b, c, t, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs = upsample_nearest3d_free_interpolate(inputs, scale_factor=2.0)
            inputs = inputs.reshape(b, t, c, *inputs.shape[2:]).permute(0, 2, 1, 3, 4)

        b, c, t, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        inputs = self.conv(inputs)
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)

        return inputs


def upfirdn2d_native(
    tensor: ms.Tensor,
    kernel: ms.Tensor,
    up: int = 1,
    down: int = 1,
    pad: Tuple[int, int] = (0, 0),
) -> ms.Tensor:
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]

    _, channel, in_h, in_w = tensor.shape
    tensor = tensor.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = tensor.shape
    kernel_h, kernel_w = kernel.shape

    out = tensor.view(-1, in_h, 1, in_w, 1, minor)
    out = ops.Pad(paddings=((0, 0), (0, 0), (0, up_y - 1), (0, 0), (0, up_x - 1), (0, 0)))(out)
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = ops.Pad(paddings=((0, 0), (max(pad_y0, 0), max(pad_y1, 0)), (max(pad_x0, 0), max(pad_x1, 0)), (0, 0)))(out)
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = ops.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = ops.conv2d(out, w.to(out.dtype))
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


def upsample_2d(
    hidden_states: ms.Tensor,
    kernel: Optional[ms.Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> ms.Tensor:
    r"""Upsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is
    a: multiple of the upsampling factor.

    Args:
        hidden_states (`ms.Tensor`):
            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel (`ms.Tensor`, *optional*):
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to nearest-neighbor upsampling.
        factor (`int`, *optional*, default to `2`):
            Integer upsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output (`ms.Tensor`):
            Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = ms.Tensor(kernel, dtype=ms.float32)
    if kernel.ndim == 1:
        kernel = ops.outer(kernel, kernel)
    kernel /= ops.sum(kernel)

    kernel = kernel * (gain * (factor**2))
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel,
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output
