#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#include <cmath>

using namespace at;  // temporal fix for pytorch<=0.4.1 (see #9848)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
        i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024  // 32 * 32
#define WARP_SIZE 32
#define THREADS_PER_PIXEL 32
#define MAX_SHARED_MEMORY 49152
#define MAX_SHARED_SCALAR_T 6144  // 49152 / 8 = 6144
#define kTileDim 32
#define kBlockRows 8
#define FORWARD_WARP_SIZE 32
#define FORWARD_THREADS_PER_PIXEL 32
#define FULL_MASK 0xffffffff

inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

__device__ inline int Loc2Index(const int n, const int c, const int h,
                                const int w, const int channel_num,
                                const int height, const int width) {
    int index = w + (h + (c + n * channel_num) * height) * width;
    return index;
}
/* TODO: move this to a common place */
template <typename scalar_t>
__device__ inline scalar_t min(scalar_t a, scalar_t b) {
    return a < b ? a : b;
}

template <typename scalar_t>
__device__ inline scalar_t max(scalar_t a, scalar_t b) {
    return a > b ? a : b;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// Splits the original matrix into submatrices with size 32 * 32.
// Each block transposes one submatrix by loading it into shared memory.
// Reference https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
template <typename scalar_t>
__global__ void BatchTranspose2DCUDAKernel(const int N, const int H,
                                           const int W, const int dh,
                                           const int dw,
                                           const scalar_t *__restrict__ X,
                                           scalar_t *__restrict__ Y) {
    __shared__ scalar_t tile[kTileDim][kTileDim + 1];
    const int n = blockIdx.x / (dh * dw);
    const int k = blockIdx.x % (dh * dw);
    const int r = k / dw;
    const int c = k % dw;
    const int offset = n * H * W;
    int x = c * kTileDim + threadIdx.x;
    int y = r * kTileDim + threadIdx.y;
    int i;
    if (x < W) {
        for (i = 0; threadIdx.y + i < kTileDim && y + i < H; i += kBlockRows) {
            tile[threadIdx.y + i][threadIdx.x] = X[offset + (y + i) * W + x];
        }
    }
    __syncthreads();
    x = r * kTileDim + threadIdx.x;
    y = c * kTileDim + threadIdx.y;
    if (x < H) {
        for (i = 0; threadIdx.y + i < kTileDim && y + i < W; i += kBlockRows) {
            Y[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

template <typename scalar_t>
__global__ void DDFForward(const int num_kernels, const scalar_t *__restrict__ bottom_data,
                           const scalar_t *__restrict__ bottom_channel_filter,
                           const scalar_t *__restrict__ bottom_spatial_filter,
                           const int kernel_size, const int dilation,
                           const int stride, const int channels,
                           const int bottom_height, const int bottom_width,
                           const int top_height, const int top_width,
                           scalar_t *__restrict__ top_data) {
    __shared__ scalar_t shared_spatial_filter[MAX_SHARED_SCALAR_T];

    bool valid_index = false;
    int index = threadIdx.x + blockIdx.y * blockDim.x;
    if (index > num_kernels - 1){
        return;
    }

    const int pixel_id = threadIdx.x / FORWARD_THREADS_PER_PIXEL; // pixel in block from 0 to 15
    const int split_id = threadIdx.x % FORWARD_THREADS_PER_PIXEL; // thread in pixel from 0 to 63
    // (n, c, ph, pw) is an element in the bottom_data
    index = index / FORWARD_THREADS_PER_PIXEL;
    const int pw = index % top_width;
    const int ph = index / top_width;
    const int n = blockIdx.x;

    const int start_w = pw * stride - ((kernel_size - 1) / 2)*dilation;
    const int end_w = pw * stride + ((kernel_size - 1) / 2)*dilation + 1;
    const int start_h = ph * stride - ((kernel_size - 1) / 2)*dilation;
    const int end_h = ph * stride + ((kernel_size - 1) / 2)*dilation + 1;

    scalar_t output_val = 0;
    scalar_t lost = 0;
    scalar_t t = 0;
    scalar_t input = 0;

    int c, spatial_filter_id, channel_filter_id, iy, ix, kernel_iy, kernel_ix, filter_c, bottom_id, top_id;

    for (c = split_id; c < kernel_size*kernel_size; c += FORWARD_THREADS_PER_PIXEL) {
        spatial_filter_id = Loc2Index(n, c, ph, pw, kernel_size * kernel_size, top_height, top_width);
        shared_spatial_filter[c * FORWARD_WARP_SIZE + pixel_id] = bottom_spatial_filter[spatial_filter_id];
    }
    __syncthreads();

    #pragma unroll
    for (c = split_id; c < channels; c += FORWARD_THREADS_PER_PIXEL) {
        output_val = 0;
        lost = 0;
        t = 0;
        input = 0;
        #pragma unroll
        for (iy = start_h; iy < end_h; iy+=dilation) {
            #pragma unroll
            for (ix = start_w; ix < end_w; ix+=dilation) {
                if (iy < 0 || iy > bottom_height - 1 || ix < 0 || ix > bottom_width - 1) {
                    continue;
                }
                kernel_iy = (iy - start_h) / dilation;
                kernel_ix = (ix - start_w) / dilation;
                filter_c = kernel_iy * kernel_size + kernel_ix;
                bottom_id = Loc2Index(n, c, iy, ix, channels, bottom_height, bottom_width);

                spatial_filter_id = Loc2Index(n, filter_c, ph, pw, kernel_size * kernel_size, top_height, top_width);
                channel_filter_id = (n * channels + c ) * kernel_size * kernel_size + filter_c;

                // Kahan and Babuska summation, Neumaier variant
                input = bottom_data[bottom_id] *
                        (shared_spatial_filter[filter_c * FORWARD_WARP_SIZE + pixel_id] +
                        bottom_channel_filter[channel_filter_id]);

                t = output_val + input;
                lost += fabs(output_val) >= fabs(input) ? (output_val - t) + input
                                                : (input - t) + output_val;
                output_val = t;
            }
        }

        top_id = Loc2Index(n, c, ph, pw, channels, top_height, top_width);
        // Kahan and Babuska summation, Neumaier variant
        top_data[top_id] = output_val + lost;
    }
}

int DDFAddForwardLauncher(const at::Tensor features, const at::Tensor channel_filter,
                          const at::Tensor spatial_filter, const int kernel_size,
                          const int dilation, const int stride,
                          const int batch_size,const int channels,
                          const int bottom_height, const int bottom_width,
                          const int top_height, const int top_width,
                          at::Tensor output){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.type(), "DDFForward", ([&] {
            const int num_kernels = top_height * top_width * FORWARD_THREADS_PER_PIXEL;
            dim3 grid(batch_size, at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK));
            const scalar_t *bottom_data = features.data<scalar_t>();
            const scalar_t *bottom_channel_filter = channel_filter.data<scalar_t>();
            const scalar_t *bottom_spatial_filter = spatial_filter.data<scalar_t>();
            scalar_t *top_data = output.data<scalar_t>();
            DDFForward<scalar_t>
                <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, bottom_data, bottom_channel_filter,
                bottom_spatial_filter, kernel_size, dilation, stride,
                channels, bottom_height, bottom_width, top_height,
                top_width, top_data);
    }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 1;
}

template <typename scalar_t>
__global__ void DDFBackward_Feature(const int num_kernels, const scalar_t *__restrict__ top_diff,
                                    const scalar_t *__restrict__ bottom_spatial_filter,
                                    const scalar_t *__restrict__ bottom_channel_filter,
                                    const int kernel_size, const int dilation,
                                    const int stride, const int channels,
                                    const int top_height, const int top_width,
                                    const int bottom_height, const int bottom_width,
                                    scalar_t *__restrict__ bottom_diff){
    __shared__ scalar_t shared_spatial_filter[MAX_SHARED_SCALAR_T];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index > num_kernels - 1) {
        return;
    }

    const int pixel_id = threadIdx.x / THREADS_PER_PIXEL;
    const int split_id = threadIdx.x % THREADS_PER_PIXEL;
    // (n, c, ph, pw) is an element in the bottom_data
    index = index / THREADS_PER_PIXEL;
    const int pw = index % bottom_width;
    const int ph = (index / bottom_width) % bottom_height;
    const int n = index / bottom_width / bottom_height;

    const int start_w = pw - ((kernel_size - 1) / 2)*dilation;
    const int end_w = pw + ((kernel_size - 1) / 2)*dilation + 1;
    const int start_h = ph - ((kernel_size - 1) / 2)*dilation;
    const int end_h = ph + ((kernel_size - 1) / 2)*dilation + 1;

    for (int c = split_id; c < kernel_size * kernel_size; c += THREADS_PER_PIXEL) {
        const int kernel_ix = c % kernel_size ;
        const int kernel_iy = c / kernel_size;
        const int ix = start_w + kernel_ix * dilation;
        const int iy = start_h + kernel_iy * dilation;
        if (ix % stride !=0 || iy % stride !=0 ||
            iy/stride < 0 || iy/stride > top_height - 1 ||
            ix/stride < 0 || ix/stride > top_width - 1){
            shared_spatial_filter[c * WARP_SIZE + pixel_id] = 0;
            continue;
        };
        const int spatial_filter_c = kernel_size * kernel_size - c - 1;
        int spatial_filter_id =
            Loc2Index(n, spatial_filter_c, iy/stride, ix/stride,
            kernel_size * kernel_size, top_height, top_width);
        shared_spatial_filter[c * WARP_SIZE + pixel_id] = bottom_spatial_filter[spatial_filter_id];
    }
    __syncthreads();

    scalar_t output_val = 0;
    scalar_t lost = 0;
    scalar_t t = 0;
    scalar_t input = 0;

    int bottom_iy, bottom_ix, iy, ix, kernel_iy, kernel_ix,
        spatial_filter_c, channel_filter_id, top_id, bottom_id;

    #pragma unroll
    for (int c = split_id; c < channels; c += THREADS_PER_PIXEL){
        output_val = 0;
        lost = 0;
        t = 0;
        input = 0;
        #pragma unroll
        for (bottom_iy = start_h; bottom_iy < end_h; bottom_iy+=dilation){
            #pragma unroll
            for (bottom_ix = start_w; bottom_ix < end_w; bottom_ix+=dilation){
                if (bottom_iy % stride != 0 || bottom_ix % stride != 0){
                    continue;
                }
                iy = bottom_iy / stride;
                ix = bottom_ix / stride;
                if (iy < 0 || iy > top_height - 1 || ix < 0 || ix > top_width - 1){
                    continue;
                }
                kernel_iy = (bottom_iy - start_h) / dilation;
                kernel_ix = (bottom_ix - start_w) / dilation;
                spatial_filter_c = kernel_iy * kernel_size + kernel_ix;
                channel_filter_id = Loc2Index(n, c, kernel_size - kernel_iy - 1,
                    kernel_size - kernel_ix - 1, channels, kernel_size, kernel_size);
                top_id = Loc2Index(n, iy, ix, c, top_height, top_width, channels);
                input = (shared_spatial_filter[spatial_filter_c * WARP_SIZE + pixel_id] +
                    bottom_channel_filter[channel_filter_id]) * top_diff[top_id];
                t = output_val + input;
                lost += fabs(output_val) >= fabs(input) ? (output_val - t) + input
                                                        : (input - t) + output_val;
                output_val = t;
            }
        }
        bottom_id = Loc2Index(n, ph, pw, c, bottom_height, bottom_width, channels);
        bottom_diff[bottom_id] = output_val + lost;
    }
}


template <typename scalar_t>
__global__ void DDFBackward_Spatial_Filter(const int num_kernels,
                                           const scalar_t *__restrict__ top_diff,
                                           const scalar_t *__restrict__ bottom_data,
                                           const int kernel_size, const int dilation,
                                           const int stride,const int channels,
                                           const int top_height, const int top_width,
                                           const int bottom_height, const int bottom_width,
                                           scalar_t *__restrict__ spatial_filter_diff) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index > num_kernels - 1) {
        return;
    }
    const int spatial_filter_channels = kernel_size * kernel_size;
    const int lane_id = index % WARP_SIZE;
    index = index / WARP_SIZE;
    const int spatial_filter_c = index % spatial_filter_channels;
    // (n, c, ph, pw) is an element in the bottom_data
    index = index / spatial_filter_channels;
    const int pw = index % top_width;
    const int ph = (index / top_width) % top_height;
    const int n = index / top_width / top_height;

    const int kernel_ix = spatial_filter_c % kernel_size;
    const int kernel_iy = spatial_filter_c / kernel_size;

    const int offset_ix = (kernel_ix - (kernel_size - 1) / 2) * dilation;
    const int offset_iy = (kernel_iy - (kernel_size - 1) / 2) * dilation;

    const int ix = pw * stride + offset_ix;
    const int iy = ph * stride + offset_iy;

    scalar_t output_val = 0;
    scalar_t lost = 0;
    scalar_t t = 0;
    scalar_t input = 0;

    int c, bottom_id, top_id, channel_filter_id;

    if (iy >= 0 && iy <= bottom_height - 1 && ix >= 0 && ix <= bottom_width - 1) {
    for (c = lane_id; c < channels; c += WARP_SIZE) {
        bottom_id =
            Loc2Index(n, c, iy, ix, channels, bottom_height, bottom_width);
        top_id = Loc2Index(n, ph, pw, c, top_height, top_width, channels);
        channel_filter_id = Loc2Index(n, c, kernel_iy, kernel_ix,
            channels, kernel_size, kernel_size);
        input = top_diff[top_id] * bottom_data[bottom_id];
        t = output_val + input;
        lost += fabs(output_val) >= fabs(input) ? (output_val - t) + input
                                              : (input - t) + output_val;
        output_val = t;
        }
    }
    __syncwarp();
    output_val = warpReduceSum(output_val);
    lost = warpReduceSum(lost);
    if (lane_id == 0) {
        const int spatial_filter_id =
            Loc2Index(n, ph, pw, spatial_filter_c, top_height, top_width, spatial_filter_channels);
        spatial_filter_diff[spatial_filter_id] = output_val + lost;
    }
}


template <typename scalar_t>
__global__ void DDFBackward_Channel_Filter(const int num_kernels,
                                           const scalar_t *__restrict__ top_diff,
                                           const scalar_t *__restrict__ bottom_data,
                                           const int kernel_size, const int dilation,
                                           const int stride, const int channels,
                                           const int top_height, const int top_width,
                                           const int bottom_height, const int bottom_width,
                                           scalar_t *__restrict__ channel_filter_diff){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index > num_kernels - 1) {
        return;
    }
    const int lane_id = index % WARP_SIZE;
    index = index / WARP_SIZE;
    const int kernel_ix = index % kernel_size;
    const int kernel_iy = (index / kernel_size) % kernel_size;
    const int c = (index / kernel_size / kernel_size ) % channels;
    const int n = index / kernel_size / kernel_size / channels;

    const int spatial_filter_c = kernel_iy * kernel_size + kernel_ix;

    const int offset_ix = (kernel_ix - (kernel_size - 1) / 2) * dilation;
    const int offset_iy = (kernel_iy - (kernel_size - 1) / 2) * dilation;

    scalar_t output_val = 0;
    scalar_t lost = 0;
    scalar_t t = 0;
    scalar_t input = 0;

    int iy, ix, bottom_iy, bottom_ix, top_id, spatial_filter_id, bottom_id;

    #pragma unroll
    for (index = lane_id; index < top_height*top_width; index+=WARP_SIZE){
        iy = index / top_width;
        ix = index % top_width;
        bottom_iy = iy * stride;
        bottom_ix = ix * stride;
        if (bottom_iy + offset_iy < 0 || bottom_iy + offset_iy > bottom_height - 1 ||
            bottom_ix + offset_ix < 0 || bottom_ix + offset_ix > bottom_width - 1){
            continue;
        }
        top_id = Loc2Index(n, c, iy, ix, channels, top_height, top_width);
        spatial_filter_id = Loc2Index(n, spatial_filter_c, iy, ix,
            kernel_size * kernel_size, top_height, top_width);
        bottom_id = Loc2Index(n, c, bottom_iy + offset_iy, bottom_ix + offset_ix, channels,
             bottom_height, bottom_width);
        input = top_diff[top_id] * bottom_data[bottom_id];
        t = output_val + input;
        lost += fabs(output_val) >= fabs(input) ? (output_val - t) + input
                                                  : (input - t) + output_val;
        output_val = t;
    }
    __syncwarp();
    output_val = warpReduceSum(output_val);
    lost = warpReduceSum(lost);
    if (lane_id == 0) {
        const int channel_filter_id = Loc2Index(n, c, kernel_iy, kernel_ix,
            channels, kernel_size, kernel_size);
        channel_filter_diff[channel_filter_id] = output_val + lost;
    }
}


int DDFAddBackwardLauncher(const at::Tensor top_grad, const at::Tensor features,
                           const at::Tensor channel_filter, const at::Tensor spatial_filter,
                           const int kernel_size, const int dilation, const int stride,
                           const int batch_size, const int channels,
                           const int top_height, const int top_width,
                           const int bottom_height, const int bottom_width,
                           at::Tensor rtop_grad, at::Tensor rbottom_grad,
                           at::Tensor rspatial_filter_grad, at::Tensor bottom_grad,
                           at::Tensor channel_filter_grad, at::Tensor spatial_filter_grad){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    top_grad.type(), "NCHW2NHWC_Top_Grad", ([&] {
        const scalar_t *bottom_data = top_grad.data<scalar_t>();
        scalar_t *top_data = rtop_grad.data<scalar_t>();
        const int dh = divideUP(channels, kTileDim);
        const int dw = divideUP(top_height * top_width, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, channels, top_height * top_width, dh, dw,
                bottom_data, top_data);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        top_grad.type(), "DDFBackward_Feature", ([&] {
        const int num_kernels =
            batch_size * bottom_height * bottom_width * THREADS_PER_PIXEL;
        const scalar_t *top_diff = rtop_grad.data<scalar_t>();
        const scalar_t *bottom_spatial_filter = spatial_filter.data<scalar_t>();
        const scalar_t *bottom_channel_filter = channel_filter.data<scalar_t>();
        scalar_t *bottom_diff = rbottom_grad.data<scalar_t>();

        DDFBackward_Feature<scalar_t>
            <<<at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK),
                THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, top_diff, bottom_spatial_filter,
                bottom_channel_filter, kernel_size, dilation,
                stride, channels, top_height, top_width,
                bottom_height, bottom_width, bottom_diff);
        }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        top_grad.type(), "NHWC2NCHW_Bottom_Grad", ([&] {
            const scalar_t *bottom_data = rbottom_grad.data<scalar_t>();
            scalar_t *top_data = bottom_grad.data<scalar_t>();
            const int dh = divideUP(bottom_height * bottom_width, kTileDim);
            const int dw = divideUP(channels, kTileDim);
            BatchTranspose2DCUDAKernel<scalar_t>
                <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                    batch_size, bottom_height * bottom_width, channels, dh, dw,
                    bottom_data, top_data);
    }));

    AT_DISPATCH_FLOATING_TYPES(
        top_grad.type(), "DDFBackward_Spatial_Filter", ([&] {
            const int num_kernels = batch_size * top_height * top_width *
                                    kernel_size * kernel_size * WARP_SIZE;
            const scalar_t *top_diff = rtop_grad.data<scalar_t>();
            const scalar_t *bottom_data = features.data<scalar_t>();
            scalar_t *spatial_filter_diff = rspatial_filter_grad.data<scalar_t>();

            DDFBackward_Spatial_Filter<scalar_t>
                <<<at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK),
                    THREADS_PER_BLOCK, 0, stream>>>(
                    num_kernels, top_diff, bottom_data,
                    kernel_size, dilation, stride, channels, top_height, top_width,
                    bottom_height, bottom_width, spatial_filter_diff);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        top_grad.type(), "NHWC2NCHW_Spatial_Filter", ([&] {
        const scalar_t *bottom_data = rspatial_filter_grad.data<scalar_t>();
        scalar_t *top_data = spatial_filter_grad.data<scalar_t>();
        const int dh = divideUP(top_height * top_width, kTileDim);
        const int dw = divideUP(kernel_size * kernel_size, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, top_height * top_width, kernel_size * kernel_size, dh, dw,
                bottom_data, top_data);
    }));

    AT_DISPATCH_FLOATING_TYPES(
        top_grad.type(), "DDFBackward_Channel_Filter", ([&] {
            const int num_kernels = batch_size * channels *
                kernel_size * kernel_size * WARP_SIZE;
            const scalar_t *top_diff = top_grad.data<scalar_t>();
            const scalar_t *bottom_data = features.data<scalar_t>();
            scalar_t *channel_filter_diff = channel_filter_grad.data<scalar_t>();

            DDFBackward_Channel_Filter<scalar_t>
                <<<at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK),
                    THREADS_PER_BLOCK, 0, stream>>>(
                    num_kernels, top_diff, bottom_data,
                    kernel_size, dilation, stride, channels, top_height,
                    top_width, bottom_height, bottom_width, channel_filter_diff);
    }));

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 1;
}