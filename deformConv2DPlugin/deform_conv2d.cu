// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp

#include "deform_conv2d.h"

namespace nvinfer1 {
namespace plugin {

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); i += (blockDim.x * gridDim.x))

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

/*******************  add_bias_kernelLauncher  ***********************/
template <typename T>
__global__ void add_bias(T* x, const T* bias, int n) {
  const int bid = blockIdx.x;
  auto b = __ldg(&bias[bid]);
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x)
    x[bid * n + tid] += b;
}

// [channel, batch, H, W] x + [channel] bias
template <typename T>
void add_bias_kernelLauncher(T* x, const T* bias, int channel, int batch, int H, int W, cudaStream_t stream) {
  dim3 grid(channel);
  int n = W * H * batch;
  int blockSize = n;
  if (std::is_same<T, half>::value && (n % 2 == 0)) {
    blockSize = n / 2;
    if (blockSize > 1024)
      blockSize = 1024;
    add_bias<<<grid, blockSize, 0, stream>>>((half2*)x, (const half2*)bias, n / 2);
  } else {
    if (blockSize > 1024)
      blockSize = 1024;
    add_bias<<<grid, blockSize, 0, stream>>>(x, bias, n);
  }
}

template <typename T>
__device__ T bilinear_interpolate(const T* in, int height, int width, T h, T w) {
  if (h <= T(-1) || T(height) <= h || w <= T(-1) || T(width) <= w) {
    return T(0);
  }


  int h_low = floor((float)h);

  int w_low = floor((float)w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - T(h_low);
  T lw = w - T(w_low);
  T hh = T(1) - lh, hw = T(1) - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = __ldg(&in[h_low * width + w_low]);
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = __ldg(&in[h_low * width + w_high]);
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = __ldg(&in[h_high * width + w_low]);
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = __ldg(&in[h_high * width + w_high]);

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
__global__ void deformable_im2col_kernel(
    int n,
    const T* input_ptr,
    const T* offset_ptr,
    const T* mask_ptr,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int batch_sz,
    int n_in_channels,
    int n_offset_grps,
    int out_h,
    int out_w,
    bool use_mask,
    T* columns_ptr) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int out_b = (index / (out_w * out_h)) % batch_sz;
    const int in_c = index / (out_w * out_h * batch_sz);
    const int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    const int grp_idx = in_c / c_per_offset_grp;

    columns_ptr += (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) + out_y * out_w + out_x);

    input_ptr += (out_b * (n_in_channels * height * width) + in_c * (height * width));

    offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;

    if (use_mask) {
      mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w * out_h * out_w;
    }

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const int mask_idx = i * weight_w + j;
        const int offset_idx = 2 * mask_idx;

        T mask_value = 1;
        if (use_mask) {
          mask_value = __ldg(&mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x]);
        }

        const T offset_h = __ldg(&offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x]);
        const T offset_w = __ldg(&offset_ptr[(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x]);
        const T y = T(out_y * stride_h - pad_h) + T(i * dilation_h) + offset_h;
        const T x = T(out_x * stride_w - pad_w) + T(j * dilation_w) + offset_w;
        *columns_ptr = mask_value * bilinear_interpolate(input_ptr, height, width, y, x);
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}

// input, weight, output are row-major
template <typename T>
void gemm(
    T* C,
    const T* A,
    const T* B,
    const int m,
    const int n,
    const int k,
    const int lda,
    const int ldb,
    const int ldc,
    cublasOperation_t trans_a,
    cublasOperation_t trans_b,
    cublasHandle_t cublas_handle,
    float scale = 1.0f) {
  cudaDataType_t Atype, Btype, Ctype, computeType;
  float alpha_float = scale;
  float beta_float = 0.0f;
  half alpha_half = half(scale);
  half beta_half = half(0.0f);
  void *alpha, *beta;
  int cublasAlgo;

  if (std::is_same<T, float>::value) {
    computeType = CUDA_R_32F;
    Atype = CUDA_R_32F;
    Btype = CUDA_R_32F;
    Ctype = CUDA_R_32F;
    alpha = &alpha_float;
    beta = &beta_float;
    cublasAlgo = CUBLAS_GEMM_DEFAULT;
  } else {
    computeType = CUDA_R_16F;
    Atype = CUDA_R_16F;
    Btype = CUDA_R_16F;
    Ctype = CUDA_R_16F;
    alpha = &alpha_half;
    beta = &beta_half;
    cublasAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }
  cublasGemmEx(
      cublas_handle,
      trans_a,
      trans_b,
      m,
      n,
      k,
      alpha,
      A,
      Atype,
      lda,
      B,
      Btype,
      ldb,
      beta,
      C,
      Ctype,
      ldc,
      computeType,
      static_cast<cublasGemmAlgo_t>(cublasAlgo));
}

int DeformConv2D::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) {
  auto input_shape = inputDesc[0].dims;
  auto bs = input_shape.d[0];
  auto in_c = input_shape.d[1];
  int in_h = input_shape.d[2];
  int in_w = input_shape.d[3];

  auto output_shape = outputDesc[0].dims;
  int out_c = output_shape.d[1];
  int out_h = output_shape.d[2];
  int out_w = output_shape.d[3];

  auto input = inputs[0];
  auto offset = inputs[1];
  auto mask = inputs[2];
  auto output = outputs[0];

  int num_kernels = in_c * bs * out_h * out_w;
  const unsigned int threads = 512;
  const unsigned int blocks = (num_kernels + threads - 1) / threads;

  if (!mColumnDev) {
    auto size = in_c * kernel_h_ * kernel_w_ * bs * out_h * out_w;
    gLogVerbose << "initialize enqueue mColumnDev count: " << size << std::endl;
    CUASSERT(cudaMalloc(&mColumnDev, size * mParamWordsize))
    CUASSERT(cudaMemset(mColumnDev, 0, size * mParamWordsize))
  }

  if (inputDesc[0].type == nvinfer1::DataType::kHALF) {
    deformable_im2col_kernel<<<blocks, threads, 0, stream>>>(
        num_kernels,
        (const half*)input,
        (const half*)offset,
        (const half*)mask,
        in_h,
        in_w,
        kernel_h_,
        kernel_w_,
        pad_h_,
        pad_w_,
        stride_h_,
        stride_w_,
        dilation_h_,
        dilation_w_,
        bs,
        in_c_,
        offset_groups_,
        out_h,
        out_w,
        use_mask_,
        (half*)mColumnDev);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
    }

    int m = out_c;
    int n = bs * out_h * out_w;
    int k = in_c * kernel_h_ * kernel_w_;
    gemm(
        (half*)output, (half*)mColumnDev, (half*)mWeightDev.get(), n, m, k, n, k, n, CUBLAS_OP_N, CUBLAS_OP_N, mCublas);

    cudaError_t gemm_err = cudaGetLastError();
    if (gemm_err != cudaSuccess) {
      printf("error in cublasSgemm_v2: %s\n", cudaGetErrorString(gemm_err));
    }

    // output [out_c, bs, out_h, out_w]
    add_bias_kernelLauncher((half*)mColumnDev, (const half*)mBiasDev.get(), out_c, bs, out_h, out_w, stream);
    cudaError_t bias_err = cudaGetLastError();
    if (bias_err != cudaSuccess) {
      printf("error in add_bias_kernelLauncher: %s\n", cudaGetErrorString(bias_err));
    }

  } else {
    // float
    deformable_im2col_kernel<<<blocks, threads, 0, stream>>>(
        num_kernels,
        (const float*)input,
        (const float*)offset,
        (const float*)mask,
        in_h,
        in_w,
        kernel_h_,
        kernel_w_,
        pad_h_,
        pad_w_,
        stride_h_,
        stride_w_,
        dilation_h_,
        dilation_w_,
        bs,
        in_c_,
        offset_groups_,
        out_h,
        out_w,
        use_mask_,
        (float*)mColumnDev);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
    }

    int m = out_c;
    int n = bs * out_h * out_w;
    int k = in_c * kernel_h_ * kernel_w_;

    // in_c_ * kernel_h_ * kernel_w_ * max_batch_size * out_h * out_w;
    gemm(
        (float*)output,
        (float*)mColumnDev,
        (float*)mWeightDev.get(),
        n,
        m,
        k,
        n,
        k,
        n,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        mCublas);
    cudaError_t gemm_err = cudaGetLastError();
    if (gemm_err != cudaSuccess) {
      printf("error in cublasSgemm_v2: %s\n", cudaGetErrorString(gemm_err));
    }

    // output [out_c, bs, out_h, out_w]
    add_bias_kernelLauncher((float*)mColumnDev, (const float*)mBiasDev.get(), out_c, bs, out_h, out_w, stream);
    cudaError_t bias_err = cudaGetLastError();
    if (bias_err != cudaSuccess) {
      printf("error in add_bias_kernelLauncher: %s\n", cudaGetErrorString(bias_err));
    }
  }

  return 0;
}

} // namespace plugin
} // namespace nvinfer1
