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
#define MAX_KS 4
#define DATA_TILE 16
#define CHANNEL_THREADS 4
#define CHANNEL_BLOCKS 8
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
__global__ void DDFForward(const scalar_t *__restrict__ bottom_data,
                           const scalar_t *__restrict__ bottom_channel_filter,
                           const scalar_t *__restrict__ bottom_spatial_filter,
                           const int kernel_size, const int dilation,
                           const int stride, const int padding,
                           const int batch_size, const int channels,
                           const int top_TileDim,
                           const int bottom_height, const int bottom_width,
                           const int top_height, const int top_width,
                           scalar_t *__restrict__ top_data){
    __shared__ scalar_t shared_spatial_filter[DATA_TILE * DATA_TILE * MAX_KS * MAX_KS];
    __shared__ scalar_t shared_channel_filter[CHANNEL_THREADS * MAX_KS * MAX_KS];
    __shared__ scalar_t shared_data[CHANNEL_THREADS * DATA_TILE * DATA_TILE];

    // current batch we're working on
    const int b = blockIdx.z / CHANNEL_BLOCKS;
    const int cb_id = blockIdx.z % CHANNEL_BLOCKS;
    bool valid_index = false;
    // calculate coordinates
    int top_tile_y = -999999;
    int top_tile_x = -999999;
    int top_y = -999999;
    int top_x = -999999;

    // the generated top_tile_y and top_tile_x must smaller than top_TileDim
    if((threadIdx.y - padding) % stride == 0 && (threadIdx.x - padding) % stride == 0){
        top_tile_y = (threadIdx.y - padding) / stride;
        top_tile_x = (threadIdx.x - padding) / stride;
    }
    if(top_tile_x >=0 && top_tile_y >=0 &&
        top_tile_x < top_TileDim &&
        top_tile_y < top_TileDim){
        valid_index=true;
        top_y = blockIdx.y * top_TileDim + top_tile_y;
        top_x = blockIdx.x * top_TileDim + top_tile_x;
    }
    // start_x = (top_tile_x * stride - padding) + padding as we need start from zero
    const int start_x = top_tile_x * stride;
    const int end_x = start_x + 2 * padding + 1;
    // start_y = (top_tile_y * stride - padding) + padding as we need start from zero
    const int start_y = top_tile_y * stride;
    const int end_y = start_y + 2 * padding + 1;

    const int bottom_x = blockIdx.x * top_TileDim * stride - padding + threadIdx.x;
    const int bottom_y = blockIdx.y * top_TileDim * stride - padding + threadIdx.y;

    // assert whether current point is a valid top_tile_x and top_tile_y
    if(valid_index){
        if (top_x < top_width && top_y < top_height){
            // load filters
            for (int i = threadIdx.z; i < kernel_size*kernel_size; i += CHANNEL_THREADS){
                int spatial_filter_id = Loc2Index(b, i, top_y, top_x, kernel_size * kernel_size, top_height, top_width);
                shared_spatial_filter[(top_tile_y * DATA_TILE + top_tile_x) * kernel_size * kernel_size + i] =
                    bottom_spatial_filter[spatial_filter_id];
            }
        }else{
            for (int i = threadIdx.z; i < kernel_size*kernel_size; i += CHANNEL_THREADS){
                shared_spatial_filter[(top_tile_y * DATA_TILE + top_tile_x) * kernel_size * kernel_size + i] = 0;
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int c = cb_id * CHANNEL_THREADS  + threadIdx.z; c < channels; c += CHANNEL_BLOCKS * CHANNEL_THREADS) {
        __syncthreads();
        //load channel filter
        if (threadIdx.x < kernel_size && threadIdx.y < kernel_size){
            int channel_filter_id = ((b * channels + c ) * kernel_size +
                threadIdx.y)* kernel_size + threadIdx.x;
            shared_channel_filter[(threadIdx.z * kernel_size + threadIdx.y) * kernel_size + threadIdx.x] =
                bottom_channel_filter[channel_filter_id];
        }

        //load data
        if(bottom_x >= 0 && bottom_x < bottom_width && bottom_y >=0 && bottom_y < bottom_height){
            int id = Loc2Index(b, c, bottom_y, bottom_x, channels, bottom_height, bottom_width);
            shared_data[(threadIdx.z * DATA_TILE + threadIdx.y)*DATA_TILE + threadIdx.x] = bottom_data[id];
        }else{
            shared_data[(threadIdx.z * DATA_TILE + threadIdx.y)*DATA_TILE + threadIdx.x] = 0;
        }
        __syncthreads();

        if(valid_index && top_x < top_width && top_y < top_height){
            scalar_t output_val = 0;
            scalar_t lost = 0;
            scalar_t t = 0;
            scalar_t input = 0;

            #pragma unroll
            for (int iy = start_y; iy < end_y; iy+=dilation) {
                #pragma unroll
                for (int ix = start_x; ix < end_x; ix+=dilation) {
                    int kernel_iy = (iy - start_y) / dilation;
                    int kernel_ix = (ix - start_x) / dilation;
                    int filter_c = kernel_iy * kernel_size + kernel_ix;

                    // Kahan and Babuska summation, Neumaier variant
                    input = shared_data[(threadIdx.z * DATA_TILE + iy) * DATA_TILE + ix] *
                        shared_spatial_filter[(top_tile_y * DATA_TILE + top_tile_x) *
                                                kernel_size * kernel_size + filter_c] *
                        shared_channel_filter[threadIdx.z * kernel_size * kernel_size + filter_c];

                    t = output_val + input;
                    lost += fabs(output_val) >= fabs(input) ? (output_val - t) + input
                                                        : (input - t) + output_val;
                    output_val = t;
                }
            }

            int top_id = Loc2Index(b, c, top_y, top_x, channels, top_height, top_width);
            // Kahan and Babuska summation, Neumaier variant
            top_data[top_id] = output_val + lost;
        }
    }
}

int DDFMulFasterForwardLauncher(const at::Tensor features, const at::Tensor channel_filter,
                                const at::Tensor spatial_filter, const int kernel_size,
                                const int dilation, const int stride,
                                const int batch_size,const int channels,
                                const int bottom_height, const int bottom_width,
                                const int top_height, const int top_width,
                                at::Tensor output){
    // one warp per pixel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int padding = (kernel_size - 1) * dilation / 2;
    const int top_TileDim = divideUP(DATA_TILE - padding*2, stride);
    const int blocks_x = divideUP(top_width, top_TileDim);
    const int blocks_y = divideUP(top_height, top_TileDim);
    const int blocks_z = batch_size * CHANNEL_BLOCKS;
    dim3 grid(blocks_x, blocks_y, blocks_z);
    dim3 block(DATA_TILE, DATA_TILE, CHANNEL_THREADS);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.type(), "DDFForward", ([&] {
            const scalar_t *bottom_data = features.data<scalar_t>();
            const scalar_t *bottom_channel_filter = channel_filter.data<scalar_t>();
            const scalar_t *bottom_spatial_filter = spatial_filter.data<scalar_t>();
            scalar_t *top_data = output.data<scalar_t>();
            DDFForward<scalar_t><<<grid, block, 0, stream>>>(
                bottom_data, bottom_channel_filter, bottom_spatial_filter,
                kernel_size, dilation, stride, padding, batch_size,
                channels, top_TileDim, bottom_height, bottom_width,
                top_height, top_width, top_data);
    }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 1;
}