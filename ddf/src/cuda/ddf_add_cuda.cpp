#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

int DDFAddForwardLauncher(
    const at::Tensor features, const at::Tensor channel_filter,
    const at::Tensor spatial_filter, const int kernel_size,
    const int dilation, const int stride,
    const int batch_size,const int channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    at::Tensor output);

int DDFAddBackwardLauncher(
    const at::Tensor top_grad, const at::Tensor features,
    const at::Tensor channel_filter, const at::Tensor spatial_filter,
    const int kernel_size, const int dilation, const int stride,
    const int batch_size, const int channels,
    const int top_height, const int top_width,
    const int bottom_height, const int bottom_width,
    at::Tensor rtop_grad, at::Tensor rbottom_grad,
    at::Tensor rspatial_filter_grad, at::Tensor bottom_grad,
    at::Tensor channel_filter_grad, at::Tensor spatial_filter_grad);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x)

int ddf_add_forward_cuda(
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

    DDFAddForwardLauncher(features, channel_filter, spatial_filter,
                          kernel_size, dilation, stride,
                          batch_size, channels,
                          bottom_height, bottom_width,
                          top_height, top_width,
                          output);
    return 1;
}

int ddf_add_backward_cuda(
    at::Tensor top_grad, at::Tensor features,
    at::Tensor channel_filter, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride,
    at::Tensor rtop_grad, at::Tensor rbottom_grad,
    at::Tensor rspatial_filter_grad, at::Tensor bottom_grad,
    at::Tensor channel_filter_grad, at::Tensor spatial_filter_grad){
    CHECK_INPUT(top_grad);
    CHECK_INPUT(features);
    CHECK_INPUT(channel_filter);
    CHECK_INPUT(spatial_filter);
    CHECK_INPUT(rtop_grad);
    CHECK_INPUT(rbottom_grad);
    CHECK_INPUT(rspatial_filter_grad);
    CHECK_INPUT(bottom_grad);
    CHECK_INPUT(channel_filter_grad);
    CHECK_INPUT(spatial_filter_grad);
    at::DeviceGuard guard(top_grad.device());

    const int batch_size = features.size(0);
    const int channels = features.size(1);
    const int bottom_height = features.size(2);
    const int bottom_width = features.size(3);
    const int top_height = top_grad.size(2);
    const int top_width = top_grad.size(3);

    rtop_grad.resize_({batch_size, int(top_height/stride), int(top_width/stride), channels});
    rbottom_grad.resize_({batch_size, bottom_height, bottom_width, channels});
    rspatial_filter_grad.resize_({batch_size, int(top_height/stride), int(top_width/stride), kernel_size*kernel_size});

    DDFAddBackwardLauncher(top_grad, features, channel_filter, spatial_filter,
                           kernel_size, dilation, stride, batch_size,
                           channels, top_height, top_width, bottom_height,
                           bottom_width, rtop_grad, rbottom_grad, rspatial_filter_grad,
                           bottom_grad, channel_filter_grad, spatial_filter_grad);
  return 1;
}
