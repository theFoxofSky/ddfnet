"""
Copyright 2021 Jingkai Zhou
"""
import math
import torch.nn as nn

from ddf import DDFPack
from timm.models.resnet import Bottleneck, ResNet, _cfg
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg

default_cfgs = {
    'ddf_mul_resnet50': _cfg(
        url='',
        interpolation='bicubic'),
    'ddf_mul_resnet101': _cfg(
        url='',
        interpolation='bicubic'),
    'ddf_mul_resnet152': _cfg(
        url='',
        interpolation='bicubic'),
    'ddf_add_resnet50': _cfg(
        url='',
        interpolation='bicubic'),
    'ddf_add_resnet101': _cfg(
        url='',
        interpolation='bicubic'),
    'ddf_add_resnet152': _cfg(
        url='',
        interpolation='bicubic')
}


class DDFMulBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None,
                 se_ratio=0.2):  # new args
        assert reduce_first == 1
        super(DDFMulBottleneck, self).__init__(
            inplanes, planes, stride, downsample, cardinality, base_width,
            reduce_first, dilation, first_dilation, act_layer, norm_layer,
            attn_layer, aa_layer, drop_block, drop_path)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        self.conv2 = DDFPack(width, kernel_size=3, stride=1 if use_aa else stride,
                             dilation=first_dilation, se_ratio=se_ratio, kernel_combine='mul')


class DDFAddBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None,
                 se_ratio=0.2):  # new args
        assert reduce_first == 1
        super(DDFAddBottleneck, self).__init__(
            inplanes, planes, stride, downsample, cardinality, base_width,
            reduce_first, dilation, first_dilation, act_layer, norm_layer,
            attn_layer, aa_layer, drop_block, drop_path)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        self.conv2 = DDFPack(width, kernel_size=3, stride=1 if use_aa else stride,
                             dilation=first_dilation, se_ratio=se_ratio, kernel_combine='add')


@register_model
def ddf_mul_resnet50(pretrained=False, **kwargs):
    model_args = dict(block=DDFMulBottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return build_model_with_cfg(
        ResNet, 'ddf_mul_resnet50', default_cfg=default_cfgs['ddf_mul_resnet50'],
        pretrained=pretrained, **model_args)


@register_model
def ddf_mul_resnet101(pretrained=False, **kwargs):
    model_args = dict(block=DDFMulBottleneck, layers=[3, 4, 23, 3],  **kwargs)
    return build_model_with_cfg(
        ResNet, 'ddf_mul_resnet101', default_cfg=default_cfgs['ddf_mul_resnet101'],
        pretrained=pretrained, **model_args)


@register_model
def ddf_mul_resnet152(pretrained=False, **kwargs):
    model_args = dict(block=DDFMulBottleneck, layers=[3, 8, 36, 3],  **kwargs)
    return build_model_with_cfg(
        ResNet, 'ddf_mul_resnet152', default_cfg=default_cfgs['ddf_mul_resnet152'],
        pretrained=pretrained, **model_args)


@register_model
def ddf_add_resnet50(pretrained=False, **kwargs):
    model_args = dict(block=DDFAddBottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return build_model_with_cfg(
        ResNet, 'ddf_add_resnet50', default_cfg=default_cfgs['ddf_add_resnet50'],
        pretrained=pretrained, **model_args)


@register_model
def ddf_add_resnet101(pretrained=False, **kwargs):
    model_args = dict(block=DDFAddBottleneck, layers=[3, 4, 23, 3],  **kwargs)
    return build_model_with_cfg(
        ResNet, 'ddf_add_resnet101', default_cfg=default_cfgs['ddf_add_resnet101'],
        pretrained=pretrained, **model_args)


@register_model
def ddf_add_resnet152(pretrained=False, **kwargs):
    model_args = dict(block=DDFAddBottleneck, layers=[3, 8, 36, 3],  **kwargs)
    return build_model_with_cfg(
        ResNet, 'ddf_add_resnet152', default_cfg=default_cfgs['ddf_add_resnet152'],
        pretrained=pretrained, **model_args)
