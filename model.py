from setting import *
from functools import partial
from typing import Callable, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import Tensor
import numpy as np


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数
        else:
            # 当只有project_conv时的情况
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)  # 注意有激活函数

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result


# class EfficientNetV2(nn.Module):
#     def __init__(self,
#                  model_cnf: list,
#                  num_classes: int = 1000,
#                  num_features: int = 1280,
#                  dropout_rate: float = 0.2,
#                  drop_connect_rate: float = 0.2):
#         super(EfficientNetV2, self).__init__()
#
#         for cnf in model_cnf:
#             assert len(cnf) == 8
#
#         norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
#
#         stem_filter_num = model_cnf[0][4]
#
#         self.stem = ConvBNAct(3,
#                               stem_filter_num,
#                               kernel_size=3,
#                               stride=2,
#                               norm_layer=norm_layer)  # 激活函数默认是SiLU
#
#         total_blocks = sum([i[0] for i in model_cnf])
#         block_id = 0
#         blocks = []
#         for cnf in model_cnf:
#             repeats = cnf[0]
#             op = FusedMBConv if cnf[-2] == 0 else MBConv
#             for i in range(repeats):
#                 blocks.append(op(kernel_size=cnf[1],
#                                  input_c=cnf[4] if i == 0 else cnf[5],
#                                  out_c=cnf[5],
#                                  expand_ratio=cnf[3],
#                                  stride=cnf[2] if i == 0 else 1,
#                                  se_ratio=cnf[-1],
#                                  drop_rate=drop_connect_rate * block_id / total_blocks,
#                                  norm_layer=norm_layer))
#                 block_id += 1
#         self.blocks = nn.Sequential(*blocks)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out")
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.stem(x)
#         x = self.blocks(x)
#         return x


class myEfficientNet(nn.Module):
    def __init__(self, drop_connect_rate: float = 0):
        super(myEfficientNet, self).__init__()
        model_cnf = [[2, 3, 1, 1, 24, 24, 0, 0],
                     [4, 3, 2, 4, 24, 64, 1, 0.1]]
        for cnf in model_cnf:
            assert len(cnf) == 8
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        stem_filter_num = model_cnf[0][4]
        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))
                block_id += 1
        self.blocks = nn.Sequential(*blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        self.con1x1 = nn.Conv2d(64, 64, 1, 1, padding=0)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(64, 3, 3, 1, padding=1)
        torch.nn.init.constant_(self.conv_out.bias, -4.0)

    def forward(self, input_x: Tensor) -> Tensor:
        input_x = input_x.permute([0, 3, 1, 2])  # 对tensor操作 置换维度 [8,400,400,3]->[8,3,400,400]
        input_x = (input_x - 255 / 2) / 255  # 应该是归一化操作[-0.5,0.5]
        x = self.stem(input_x)
        x = self.blocks(x)
        x = self.act(self.con1x1(x))
        x = torch.sigmoid(self.conv_out(x))

        # Prediction Offsets
        final_y = x.shape[3]  # 50
        original_shape_y = input_x.shape[3]  # 400
        final_x = x.shape[2]  # 50
        original_shape_x = input_x.shape[2]  # 400

        # 8 间隔
        x_offset, y_offset = np.meshgrid(
            np.linspace(0, original_shape_y - original_shape_y / final_y, final_y),
            np.linspace(0, original_shape_x - original_shape_x / final_x, final_x))

        x_clone = x.clone()  # to allow for in-place operations
        x_clone[:, 0, :, :] *= original_shape_x / final_x  # local offset  8
        x_clone[:, 1, :, :] *= original_shape_y / final_y  # local offset  8
        x_clone[:, 0, :, :] += torch.tensor(y_offset[np.newaxis, :, :], device=device).float() - 0.5
        x_clone[:, 1, :, :] += torch.tensor(x_offset[np.newaxis, :, :], device=device).float() - 0.5

        x_clone = x_clone.permute(0, 2, 3, 1)
        return x_clone.view(x_clone.shape[0], -1, 3)

