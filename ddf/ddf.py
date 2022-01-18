""" DDF operation and DDF/DDF-Up Pack

The official implementation of the CVPR 2021 paper:
* Decoupled Dynamic Filter Networks - https://arxiv.org/abs/2104.14107

Thanks to Jiaqi Wang for the CARAFE repository and the associated paper:
* CARAFE: Content-Aware ReAssembly of FEatures - https://arxiv.org/abs/1905.02188

Copyright 2021 Jingkai Zhou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.init import calculate_gain

from . import ddf_mul_ext, ddf_mul_faster_ext, ddf_add_ext, ddf_add_faster_ext

OP_DICT = {
    'mul': ddf_mul_ext,
    'mul_faster': ddf_mul_faster_ext,
    'add': ddf_add_ext,
    'add_faster': ddf_add_faster_ext
}


class DDFFunction(Function):
    @staticmethod
    def forward(ctx, features, channel_filter, spatial_filter,
                kernel_size=3, dilation=1, stride=1, kernel_combine='mul', version=''):
        # check args
        assert features.is_cuda, 'input feature must be a CUDA tensor.'
        assert channel_filter.is_cuda, 'channel_filter must be a CUDA tensor.'
        assert spatial_filter.is_cuda, 'spatial_filter must be a CUDA tensor.'

        # TODO: fix CUDA code to support HALF operation
        if features.dtype == torch.float16:
            features = features.float()
        if channel_filter.dtype == torch.float16:
            channel_filter = channel_filter.float()
        if spatial_filter.dtype == torch.float16:
            spatial_filter = spatial_filter.float()

        # check channel_filter size
        b, c, h, w = features.size()
        bc, cc, hc, wc = channel_filter.size()
        assert bc == b and cc == c,\
            "channel_filter size {} does not match feature size {}".format(
                channel_filter.size(), features.size())
        assert hc == kernel_size and wc == kernel_size,\
            "channel_filter size {} does not match kernel size {}".format(
                channel_filter.size(), kernel_size)

        # check spatial_filter size
        bs, cs, hs, ws, = spatial_filter.size()
        assert bs == b and hs == h // stride and ws == w // stride,\
            "spatial_filter size {} does not match feature size {} with stride {}".format(
                spatial_filter.size(), features.size(), stride)
        assert cs == kernel_size ** 2,\
            "spatial_filter size {} does not match kernel size {}".format(
                spatial_filter.size(), kernel_size)

        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1 and dilation >= 1 and stride >= 1
        assert kernel_combine in {'mul', 'add'}, \
            'only support mul or add combination, instead of {}'.format(kernel_combine)

        # record important info
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.stride = stride
        ctx.op_type = kernel_combine

        # build output tensor
        output = features.new_zeros((b, c, h//stride, w//stride))

        # choose a suitable CUDA implementation based on the input feature, filter size, and combination type.
        if version == 'f':
            op_type = kernel_combine + '_faster'
        elif version == 'o':
            op_type = kernel_combine
        elif kernel_size <= 4 and h >= 14 and w >= 14 and stride == 1:
            op_type = kernel_combine+'_faster'
        else:
            op_type = kernel_combine

        OP_DICT[op_type].forward(features, channel_filter, spatial_filter,
                                 kernel_size, dilation, stride, output)
        if features.requires_grad or channel_filter.requires_grad or spatial_filter.requires_grad:
            ctx.save_for_backward(features, channel_filter, spatial_filter)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        # TODO: support HALF operation
        if grad_output.dtype == torch.float16:
            grad_output = grad_output.float()

        kernel_size = ctx.kernel_size
        dilation = ctx.dilation
        stride = ctx.stride
        op_type = ctx.op_type

        features, channel_filter, spatial_filter = ctx.saved_tensors
        rgrad_output = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input = torch.zeros_like(features, requires_grad=False)
        rgrad_spatial_filter = torch.zeros_like(spatial_filter, requires_grad=False)
        grad_input = torch.zeros_like(features, requires_grad=False)
        grad_channel_filter = torch.zeros_like(channel_filter, requires_grad=False)
        grad_spatial_filter = torch.zeros_like(spatial_filter, requires_grad=False)

        # TODO: optimize backward CUDA code.
        OP_DICT[op_type].backward(grad_output.contiguous(), features, channel_filter,
                                  spatial_filter, kernel_size, dilation, stride,
                                  rgrad_output, rgrad_input, rgrad_spatial_filter,
                                  grad_input, grad_channel_filter, grad_spatial_filter)

        return grad_input, grad_channel_filter, grad_spatial_filter, None, None, None, None, None


ddf = DDFFunction.apply


class FilterNorm(nn.Module):
    def __init__(self, in_channels, kernel_size, filter_type,
                 nonlinearity='linear', running_std=False, running_mean=False):
        assert filter_type in ('spatial', 'channel')
        assert in_channels >= 1
        super(FilterNorm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2), requires_grad=True)

    def forward(self, x):
        if self.filter_type == 'spatial':
            b, _, h, w = x.size()
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]
        elif self.filter_type == 'channel':
            b = x.size(0)
            c = self.in_channels
            x = x.reshape(b, c, -1)
            x = x - x.mean(dim=2).reshape(b, c, 1)
            x = x / (x.std(dim=2).reshape(b, c, 1) + 1e-10)
            x = x.reshape(b, -1)
            if self.runing_std:
                x = x * self.std[None, :]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :]
        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
        return x


def build_spatial_branch(in_channels, kernel_size, head=1,
                         nonlinearity='relu', stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, head * kernel_size ** 2, 1, stride=stride),
        FilterNorm(head, kernel_size, 'spatial', nonlinearity))


def build_channel_branch(in_channels, kernel_size,
                         nonlinearity='relu', se_ratio=0.2):
    assert se_ratio > 0
    mid_channels = int(in_channels * se_ratio)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(in_channels, mid_channels, 1),
        nn.ReLU(True),
        nn.Conv2d(mid_channels, in_channels * kernel_size ** 2, 1),
        FilterNorm(in_channels, kernel_size, 'channel', nonlinearity, running_std=True))


class DDFPack(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, dilation=1, head=1,
                 se_ratio=0.2, nonlinearity='relu', kernel_combine='mul'):
        super(DDFPack, self).__init__()
        assert kernel_size > 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.head = head
        self.kernel_combine = kernel_combine

        self.spatial_branch = build_spatial_branch(
            in_channels, kernel_size, head, nonlinearity, stride)

        self.channel_branch = build_channel_branch(
            in_channels, kernel_size, nonlinearity, se_ratio)

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.head
        k = self.kernel_size
        s = self.stride
        channel_filter = self.channel_branch(x).reshape(b*g, c//g, k, k)
        spatial_filter = self.spatial_branch(x).reshape(b*g, -1, h//s, w//s)
        x = x.reshape(b*g, c//g, h, w)
        out = ddf(x, channel_filter, spatial_filter,
                  self.kernel_size, self.dilation, self.stride, self.kernel_combine)
        return out.reshape(b, c, h//s, w//s)


class DDFUpPack(nn.Module):
    def __init__(self, in_channels, kernel_size=3, scale_factor=2, dilation=1, head=1, se_ratio=0.2,
                 nonlinearity='linear', dw_kernel_size=3, joint_channels=-1, kernel_combine='mul'):
        super(DDFUpPack, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.head = head
        self.scale_factor = scale_factor
        self.kernel_combine = kernel_combine

        self.spatial_branch = nn.ModuleList()
        self.channel_branch = nn.ModuleList()

        for i in range(scale_factor ** 2):
            # build spatial branches
            if joint_channels < 1:
                dw_kernel_size = max(dw_kernel_size, 3)
                self.spatial_branch.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, dw_kernel_size,
                                  padding=kernel_size//2, groups=in_channels),
                        build_spatial_branch(
                            in_channels, kernel_size, head, nonlinearity, 1)))
            else:
                self.spatial_branch.append(
                    build_spatial_branch(
                        in_channels, kernel_size, head, nonlinearity, 1))

            self.channel_branch.append(
                build_channel_branch(
                    in_channels, kernel_size, nonlinearity, se_ratio))

    def forward(self, x, joint_x=None):
        joint_x = x if joint_x is None else joint_x
        outs = []
        b, c, h, w = x.shape
        g = self.head
        k = self.kernel_size
        _x = x.reshape(b*g, c//g, h, w)
        for s_b, c_b in zip(self.spatial_branch, self.channel_branch):
            channel_filter = c_b(x).reshape(b*g, c//g, k, k)
            spatial_filter = s_b(joint_x).reshape(b*g, -1, h, w)
            o = ddf(_x, channel_filter, spatial_filter,
                    self.kernel_size, self.dilation, 1, self.head, self.kernel_combine).type_as(x)
            outs.append(o.reshape(b, c, h, w))
        out = torch.stack(outs, dim=2)
        out = out.reshape(out.size(0), -1, out.size(-2), out.size(-1))
        return F.pixel_shuffle(out, self.scale_factor)
