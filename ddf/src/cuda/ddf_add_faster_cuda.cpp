#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

int DDFAddFasterForwardLauncher(
    const at::Tensor features, const at::Tensor channel_filter,
    const at::Tensor spatial_filter, const int kernel_size,
    const int dilation, const int stride,
    const int batch_size,const int channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    at::Tensor output);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x)

int ddf_add_faster_forward_cuda(
    at::Tensor features,at::Tensor channel_filter, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride, at::Tensor output){
    CHECK_INPUT(features);
    CHECK_INPUT(channel_filter);
    CHECK_INPUT(spatial_filter);
    CHECK_INPUT(output);
    at::DeviceGuard guard(features.device());

    const int batch_size = features.size(0);
    const int channels = features.size(1);
    const int bottom_height = features.size(2);
    const int bottom_width = features.size(3);
    const int top_height = output.size(2);
    const int top_width = output.size(3);

    DDFAddFasterForwardLauncher(features, channel_filter, spatial_filter,
                                kernel_size, dilation, stride,
                                batch_size, channels,
                                bottom_height, bottom_width,
                                top_height, top_width,
                                output);
    return 1;
}
