"""
Copyright 2021 Jingkai Zhou
"""
import time
import torch
import torch.nn as nn

from ddf import DDFPack, ddf


def test_time(op, feat, name, num_warm_up=100, num_iters=1000):
    op.eval()
    sum_t = 0
    min_t = 999
    size = feat.size()
    with torch.no_grad():
        for i in range(num_warm_up + num_iters):
            if i < num_warm_up:
                _ = op(feat)
                continue
            st = time.time()
            torch.cuda.synchronize()
            _ = op(feat)
            torch.cuda.synchronize()
            et = time.time() - st
            sum_t += et
            if min_t > et:
                min_t = et
    print('Inference time of {} on size {} is {} ms (min {}).'.format(
        name, size, sum_t*1000/num_iters, min_t*1000))


def test_ddf_op(feat, channel, spatial, num_warm_up=100, num_iters=1000):
    sum_t = 0
    min_t = 999
    size = feat.size()
    with torch.no_grad():
        for i in range(num_warm_up + num_iters):
            if i < num_warm_up:
                _ = ddf(feat, channel, spatial, 3, 1, 1, 'mul')
                continue
            st = time.time()
            torch.cuda.synchronize()
            _ = ddf(feat, channel, spatial, 3, 1, 1, 'mul')
            torch.cuda.synchronize()
            et = time.time() - st
            sum_t += et
            if min_t > et:
                min_t = et
    print('Inference time of ddf_op on size {} is {} ms (min {}).'.format(
        size, sum_t*1000/num_iters, min_t*1000))


conv = nn.Conv2d(256, 256, 3, padding=1).cuda()
dw_conv = nn.Conv2d(256, 256, 3, padding=1,groups=256).cuda()
ddf_layer = DDFPack(256).cuda()

torch.cuda.empty_cache()
feat = torch.randn(2, 256, 200, 300).cuda()
spatial_kernel = torch.rand(2, 9, 200, 300).cuda()
channel_kernel = torch.rand(2, 256, 3, 3).cuda()
torch.cuda.empty_cache()

test_time(conv, feat, 'conv')
test_time(dw_conv, feat, 'dw_conv')
test_time(ddf_layer, feat, 'ddf_layer')
test_ddf_op(feat, channel_kernel, spatial_kernel)

print('\n ===================================== \n')

torch.cuda.empty_cache()
feat = torch.randn(32, 256, 224, 224).cuda()
spatial_kernel = torch.rand(32, 9, 224, 224).cuda()
channel_kernel = torch.rand(32, 256, 3, 3).cuda()
torch.cuda.empty_cache()

test_time(conv, feat, 'conv')
test_time(dw_conv, feat, 'dw_conv')
test_time(ddf_layer, feat, 'ddf_layer')
test_ddf_op(feat, channel_kernel, spatial_kernel)

print('\n ===================================== \n')

torch.cuda.empty_cache()
feat = torch.randn(32, 256, 112, 112).cuda()
spatial_kernel = torch.rand(32, 9, 112, 112).cuda()
channel_kernel = torch.rand(32, 256, 3, 3).cuda()
torch.cuda.empty_cache()

test_time(conv, feat, 'conv')
test_time(dw_conv, feat, 'dw_conv')
test_time(ddf_layer, feat, 'ddf_layer')
test_ddf_op(feat, channel_kernel, spatial_kernel)

print('\n ===================================== \n')

torch.cuda.empty_cache()
feat = torch.randn(32, 256, 56, 56).cuda()
spatial_kernel = torch.rand(32, 9, 56, 56).cuda()
channel_kernel = torch.rand(32, 256, 3, 3).cuda()
torch.cuda.empty_cache()

test_time(conv, feat, 'conv')
test_time(dw_conv, feat, 'dw_conv')
test_time(ddf_layer, feat, 'ddf_layer')
test_ddf_op(feat, channel_kernel, spatial_kernel)

print('\n ===================================== \n')

torch.cuda.empty_cache()
feat = torch.randn(32, 256, 28, 28).cuda()
spatial_kernel = torch.rand(32, 9, 28, 28).cuda()
channel_kernel = torch.rand(32, 256, 3, 3).cuda()
torch.cuda.empty_cache()

test_time(conv, feat, 'conv')
test_time(dw_conv, feat, 'dw_conv')
test_time(ddf_layer, feat, 'ddf_layer')
test_ddf_op(feat, channel_kernel, spatial_kernel)

print('\n ===================================== \n')

torch.cuda.empty_cache()
feat = torch.randn(32, 256, 14, 14).cuda()
spatial_kernel = torch.rand(32, 9, 14, 14).cuda()
channel_kernel = torch.rand(32, 256, 3, 3).cuda()
torch.cuda.empty_cache()

test_time(conv, feat, 'conv')
test_time(dw_conv, feat, 'dw_conv')
test_time(ddf_layer, feat, 'ddf_layer')
test_ddf_op(feat, channel_kernel, spatial_kernel)

print('\n ===================================== \n')

torch.cuda.empty_cache()
feat = torch.randn(32, 256, 7, 7).cuda()
spatial_kernel = torch.rand(32, 9, 7, 7).cuda()
channel_kernel = torch.rand(32, 256, 3, 3).cuda()
torch.cuda.empty_cache()

test_time(conv, feat, 'conv')
test_time(dw_conv, feat, 'dw_conv')
test_time(ddf_layer, feat, 'ddf_layer')
test_ddf_op(feat, channel_kernel, spatial_kernel)
