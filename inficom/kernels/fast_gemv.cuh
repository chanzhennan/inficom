#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utils.h"

#define SHARED_MEM_MAX_ROWS 64

/*
    GEMV kernel using FP16 to accumulate. Modified by @Infinigence.
*/
__global__ __forceinline__ void fast_gemv_acc_fp16_kernel(
    half* mat, half* vec, half* res, unsigned int n,
    unsigned int num_per_thread) {
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
  unsigned int start_idx = threadIdx.x;
  half2 vec_val[4];
  half2 mat_val[8];

  half2 sum[2];
  sum[0] = {__float2half(0.0f), __float2half(0.0f)};
  sum[1] = {__float2half(0.0f), __float2half(0.0f)};
  half2 gsum;

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x) << 3;
    if (j >= n) {
      break;
    }
    *(float4*)(&vec_val[0]) = *(float4*)(&vec[j]);
    *(float4*)(&mat_val[0]) = *(float4*)(&mat[row * n + j]);
    *(float4*)(&mat_val[4]) = *(float4*)(&mat[(row + 1) * n + j]);

    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0], mat_val[0]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1], mat_val[1]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[2], mat_val[2]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[3], mat_val[3]));

    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0], mat_val[4]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1], mat_val[5]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[2], mat_val[6]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[3], mat_val[7]));
  }

  gsum.x = __hadd(sum[0].x, sum[0].y);
  gsum.y = __hadd(sum[1].x, sum[1].y);

  static __shared__ half warpLevelSums[WARP_SIZE];

  gsum.x = blockReduceSum(gsum.x, warpLevelSums);
  gsum.y = blockReduceSum(gsum.y, warpLevelSums);

  if (tid == 0) {
    *(half2*)(&res[row]) = gsum;
  }
}

/*
    GEMV kernel using FP32 to accumulate. This implementation comes directly
   from https://github.com/wangsiping97/FastGEMV.
*/
__global__ __forceinline__ void fast_gemv_acc_fp32_kernel(
    half* mat, half* vec, half* res, unsigned int n,
    unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
      half2* vec_h1 = (half2*)&vec_val.x;
      half2* vec_h2 = (half2*)&vec_val.y;
      half2* vec_h3 = (half2*)&vec_val.z;
      half2* vec_h4 = (half2*)&vec_val.w;
      half2* mat_h1 = (half2*)&mat_val.x;
      half2* mat_h2 = (half2*)&mat_val.y;
      half2* mat_h3 = (half2*)&mat_val.z;
      half2* mat_h4 = (half2*)&mat_val.w;
      sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}

// __global__ __forceinline__ void fast_gemv_acc_fp16_kernel_overlap(
//                           half* mat, half* vec, half* res, unsigned int n,
//                           unsigned int num_per_thread) {

//   // each thread load num_per_thread elements from global
//   unsigned int tid = threadIdx.x;
//   unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y);
//   unsigned int start_idx = threadIdx.x;
//   half2 vec_val[4];
//   half2 mat_val[4];

//   half2 sum;
//   sum = {__float2half(0.0f), __float2half(0.0f)};
//   // sum[1] = {__float2half(0.0f), __float2half(0.0f)};
//   half gsum;

// #pragma unroll
//   for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
//     unsigned int j = (start_idx + iter * blockDim.x) << 3;
//     if (j >= n) {break;}
//     *(float4*)(&vec_val[0]) = *(float4*)(&vec[j]);
//     *(float4*)(&mat_val[0]) = *(float4*)(&mat[row * n + j]);
//     // *(float4*)(&mat_val[4]) = *(float4*)(&mat[(row + 1) * n + j]);

//     sum = __hadd2(sum, __hmul2(vec_val[0], mat_val[0]));
//     sum = __hadd2(sum, __hmul2(vec_val[1], mat_val[1]));
//     sum = __hadd2(sum, __hmul2(vec_val[2], mat_val[2]));
//     sum = __hadd2(sum, __hmul2(vec_val[3], mat_val[3]));
//   }

//   gsum = __hadd(sum.x, sum.y);
//   // gsum.y = __hadd(sum[1].x, sum[1].y);

//   static __shared__ half warpLevelSums[WARP_SIZE];

//   gsum = blockReduceSum(gsum, warpLevelSums);
//   // gsum.y = blockReduceSum(gsum.y, warpLevelSums);

//   if (tid == 0) {
//     *(half*)(&res[row]) = gsum;
//   }
// }

__global__ __forceinline__ void fast_gemv_acc_fp16_kernel_overlap(
    half* mat, half* vec, half* res, unsigned int n,
    unsigned int num_per_thread) {
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y);
  unsigned int start_idx = threadIdx.x;
  half2 vec_val[2][4];
  half2 mat_val[2][4];

  half2 sum;
  sum = {__float2half(0.0f), __float2half(0.0f)};
  // sum[1] = {__float2half(0.0f), __float2half(0.0f)};
  half gsum;

  unsigned int j = start_idx << 3;

  *(float4*)(&vec_val[0][0]) = *(float4*)(&vec[j]);
  *(float4*)(&mat_val[0][0]) = *(float4*)(&mat[row * n + j]);

  __syncthreads();

#pragma unroll
  for (int iter = 1; iter < DIV_UP(num_per_thread, 8); iter++) {
    j = (start_idx + iter * blockDim.x) << 3;
    if (j >= n) {
      break;
    }

    if (iter % 2 == 0) {
      *(float4*)(&vec_val[0][0]) = *(float4*)(&vec[j]);
      *(float4*)(&mat_val[0][0]) = *(float4*)(&mat[row * n + j]);

      sum = __hadd2(sum, __hmul2(vec_val[1][0], mat_val[1][0]));
      sum = __hadd2(sum, __hmul2(vec_val[1][1], mat_val[1][1]));
      sum = __hadd2(sum, __hmul2(vec_val[1][2], mat_val[1][2]));
      sum = __hadd2(sum, __hmul2(vec_val[1][3], mat_val[1][3]));

    } else {
      *(float4*)(&vec_val[1][0]) = *(float4*)(&vec[j]);
      *(float4*)(&mat_val[1][0]) = *(float4*)(&mat[row * n + j]);

      sum = __hadd2(sum, __hmul2(vec_val[0][0], mat_val[0][0]));
      sum = __hadd2(sum, __hmul2(vec_val[0][1], mat_val[0][1]));
      sum = __hadd2(sum, __hmul2(vec_val[0][2], mat_val[0][2]));
      sum = __hadd2(sum, __hmul2(vec_val[0][3], mat_val[0][3]));
    }
  }

  if ((DIV_UP(num_per_thread, 8)) % 2 == 0) {
    sum = __hadd2(sum, __hmul2(vec_val[1][0], mat_val[1][0]));
    sum = __hadd2(sum, __hmul2(vec_val[1][1], mat_val[1][1]));
    sum = __hadd2(sum, __hmul2(vec_val[1][2], mat_val[1][2]));
    sum = __hadd2(sum, __hmul2(vec_val[1][3], mat_val[1][3]));
  } else {
    sum = __hadd2(sum, __hmul2(vec_val[0][0], mat_val[0][0]));
    sum = __hadd2(sum, __hmul2(vec_val[0][1], mat_val[0][1]));
    sum = __hadd2(sum, __hmul2(vec_val[0][2], mat_val[0][2]));
    sum = __hadd2(sum, __hmul2(vec_val[0][3], mat_val[0][3]));
  }

  gsum = __hadd(sum.x, sum.y);
  // gsum.y = __hadd(sum[1].x, sum[1].y);

  static __shared__ half warpLevelSums[WARP_SIZE];

  gsum = blockReduceSum(gsum, warpLevelSums);
  // gsum.y = blockReduceSum(gsum.y, warpLevelSums);

  if (tid == 0) {
    *(half*)(&res[row]) = gsum;
  }
}

__global__ __forceinline__ void fast_gemv_acc_fp16_kernel_overlap_turbo(
    half* mat, half* vec, half* res, unsigned int n,
    unsigned int num_per_thread) {
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
  unsigned int start_idx = threadIdx.x;
  half2 vec_val[2][4];
  half2 mat_val[2][8];

  half2 sum[2];
  sum[0] = {__float2half(0.0f), __float2half(0.0f)};
  sum[1] = {__float2half(0.0f), __float2half(0.0f)};
  half2 gsum;

  unsigned int j = start_idx << 3;

  *(float4*)(&vec_val[0][0]) = *(float4*)(&vec[j]);
  *(float4*)(&mat_val[0][0]) = *(float4*)(&mat[row * n + j]);
  *(float4*)(&mat_val[0][4]) = *(float4*)(&mat[(row + 1) * n + j]);
  __syncthreads();

#pragma unroll
  for (int iter = 1; iter < DIV_UP(num_per_thread, 8); iter++) {
    j = (start_idx + iter * blockDim.x) << 3;
    if (j >= n) {
      break;
    }

    if (iter % 2 == 0) {
      *(float4*)(&vec_val[0][0]) = *(float4*)(&vec[j]);
      *(float4*)(&mat_val[0][0]) = *(float4*)(&mat[row * n + j]);
      *(float4*)(&mat_val[0][4]) = *(float4*)(&mat[(row + 1) * n + j]);

      sum[0] = __hadd2(sum[0], __hmul2(vec_val[1][0], mat_val[1][0]));
      sum[0] = __hadd2(sum[0], __hmul2(vec_val[1][1], mat_val[1][1]));
      sum[0] = __hadd2(sum[0], __hmul2(vec_val[1][2], mat_val[1][2]));
      sum[0] = __hadd2(sum[0], __hmul2(vec_val[1][3], mat_val[1][3]));

      sum[1] = __hadd2(sum[1], __hmul2(vec_val[1][0], mat_val[1][4]));
      sum[1] = __hadd2(sum[1], __hmul2(vec_val[1][1], mat_val[1][5]));
      sum[1] = __hadd2(sum[1], __hmul2(vec_val[1][2], mat_val[1][6]));
      sum[1] = __hadd2(sum[1], __hmul2(vec_val[1][3], mat_val[1][7]));

    } else {
      *(float4*)(&vec_val[1][0]) = *(float4*)(&vec[j]);
      *(float4*)(&mat_val[1][0]) = *(float4*)(&mat[row * n + j]);
      *(float4*)(&mat_val[1][4]) = *(float4*)(&mat[(row + 1) * n + j]);

      sum[0] = __hadd2(sum[0], __hmul2(vec_val[0][0], mat_val[0][0]));
      sum[0] = __hadd2(sum[0], __hmul2(vec_val[0][1], mat_val[0][1]));
      sum[0] = __hadd2(sum[0], __hmul2(vec_val[0][2], mat_val[0][2]));
      sum[0] = __hadd2(sum[0], __hmul2(vec_val[0][3], mat_val[0][3]));

      sum[1] = __hadd2(sum[1], __hmul2(vec_val[0][0], mat_val[0][4]));
      sum[1] = __hadd2(sum[1], __hmul2(vec_val[0][1], mat_val[0][5]));
      sum[1] = __hadd2(sum[1], __hmul2(vec_val[0][2], mat_val[0][6]));
      sum[1] = __hadd2(sum[1], __hmul2(vec_val[0][3], mat_val[0][7]));
    }
  }

  if ((DIV_UP(num_per_thread, 8)) % 2 == 0) {
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1][0], mat_val[1][0]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1][1], mat_val[1][1]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1][2], mat_val[1][2]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1][3], mat_val[1][3]));

    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1][0], mat_val[1][4]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1][1], mat_val[1][5]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1][2], mat_val[1][6]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1][3], mat_val[1][7]));
  } else {
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0][0], mat_val[0][0]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0][1], mat_val[0][1]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0][2], mat_val[0][2]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0][3], mat_val[0][3]));

    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0][0], mat_val[0][4]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0][1], mat_val[0][5]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0][2], mat_val[0][6]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0][3], mat_val[0][7]));
  }

  gsum.x = __hadd(sum[0].x, sum[0].y);
  gsum.y = __hadd(sum[1].x, sum[1].y);

  static __shared__ half warpLevelSums[WARP_SIZE];

  gsum.x = blockReduceSum(gsum.x, warpLevelSums);
  gsum.y = blockReduceSum(gsum.y, warpLevelSums);

  if (tid == 0) {
    *(half2*)(&res[row]) = gsum;
  }
}

/*
    GEMV kernel using FP16 to accumulate. With adding bias fused.
*/
__global__ void dequant_gemv_4bit_wo_bias(const uint8_t* qweight, half* vec,
                                          half* res, unsigned int n,
                                          unsigned int num_per_thread,
                                          const half* zeros, const half* scales,
                                          const int c_in, const int qc_in,
                                          const int c_out, const int gs) {
  // int mul_byte[4] = {1, 3, 1, 1};
  // int div_byte[4] = {2, 8, 4, 8};

  // half2 temp_sum;
  // temp_sum.x = __float2half(0.0f);
  // temp_sum.y = __float2half(0.0f);
  float sum = 0.0f;
  // half sum = __float2half(0.0f);
  // float sum = 0.0f;
  // each thread load num_per_thread elements from global
  int tidx = threadIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int start_idx = threadIdx.x;
  half2 vec_val[16];

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 32); iter++) {
    int j = (start_idx + iter * blockDim.x) << 5;
    if (j >= n) {
      break;
    }

    *(float4*)(&vec_val[0]) = *(float4*)(&vec[j]);
    *(float4*)(&vec_val[4]) = *(float4*)(&vec[j + 8]);
    *(float4*)(&vec_val[8]) = *(float4*)(&vec[j + 16]);
    *(float4*)(&vec_val[12]) = *(float4*)(&vec[j + 24]);

    // offset computation
    int w_offset_fp16 = j;
    int w_offset_uint8 = w_offset_fp16 / 2;
    int z_offset_uint8 = w_offset_fp16 / gs / 2;

    // loading quantized weights and zeros
    uint8_t qw[16];

    // uint8 -> half
    // half w[8], z[8];
    half w[32];
    float w_f[32];

    *(float4*)(&qw[0]) = *(float4*)(&qweight[row * qc_in + w_offset_uint8]);
    *((float4*)&w[0]) = _4bits_dequant(&qw[0]);
    *((float4*)&w[8]) = _4bits_dequant(&qw[4]);
    *((float4*)&w[16]) = _4bits_dequant(&qw[8]);
    *((float4*)&w[24]) = _4bits_dequant(&qw[12]);

    // pack zeros
    int inter_zero_idx = w_offset_fp16 / gs % 2;
    // half z_r[2] = {z[inter_zero_idx], z[inter_zero_idx]};
    half z = zeros[row * c_in / gs + j / gs];
    // half z_r[2] = {z, z};

    // load scales in fp16
    half s = scales[row * c_in / gs + j / gs];
// half s_r[2] = {s, s};

// scaling
#pragma unroll
    for (int i = 0; i < 32; i++) {
      // *(half2*)(&w[i]) = __hsub2(*(half2*)(&w[i]), *(half2*)(&z_r[0]));
      w_f[i] = __half2float(w[i]) - __half2float(z);
    }

    // half2 temp_temp_sum = __hmul2(vec_val[0], *(half2*)(&w[0]));
    // #pragma unroll
    // for (int t = 1; t < 16; t++){
    //   temp_temp_sum = __hadd2(temp_temp_sum, __hmul2(vec_val[t],
    //   *(half2*)(&w[(t << 1)])));
    // }
    // temp_sum = __hadd2(temp_sum, __hmul2(temp_temp_sum, *(half2*)(&s_r[0])));
    float temp_sum = 0.0f;

#pragma unroll
    for (int t = 0; t < 16; t++) {
      temp_sum += w_f[(t << 1)] * __half2float(vec_val[t].x);
      temp_sum += w_f[(t << 1) + 1] * __half2float(vec_val[t].y);
    }

    sum = sum + temp_sum * __half2float(s);
  }

  // sum = __hadd(temp_sum.x, temp_sum.y);
  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tidx == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0f;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tidx == 0) {
    res[row] = __float2half(sum);
  }
}

__global__ void dequant_gemv_4bit_wo_bias_lop3(
    const uint8_t* qweight, half* vec, half* res, unsigned int n,
    unsigned int num_per_thread, const half* zeros, const half* scales,
    const int c_in, const int qc_in, const int c_out, const int gs) {
  // int mul_byte[4] = {1, 3, 1, 1};
  // int div_byte[4] = {2, 8, 4, 8};

  // half2 temp_sum;
  // temp_sum.x = __float2half(0.0f);
  // temp_sum.y = __float2half(0.0f);
  float sum = 0.0f;
  // half sum = __float2half(0.0f);
  // float sum = 0.0f;
  // each thread load num_per_thread elements from global
  int tidx = threadIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int start_idx = threadIdx.x;
  half2 vec_val[16];

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOT_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t MAGIC_NUM_0 = 0x64006400;        // `1024`
  static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;        // `64`
  static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4;  // `64` >> 4

  uint32_t h1[4];
  uint32_t h2[4];
  uint32_t h3[4];
  uint32_t h4[4];
#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 32); iter++) {
    int j = (start_idx + iter * blockDim.x) << 5;
    if (j >= n) {
      break;
    }

    *(float4*)(&vec_val[0]) = *(float4*)(&vec[j]);
    *(float4*)(&vec_val[4]) = *(float4*)(&vec[j + 8]);
    *(float4*)(&vec_val[8]) = *(float4*)(&vec[j + 16]);
    *(float4*)(&vec_val[12]) = *(float4*)(&vec[j + 24]);

    // offset computation
    int w_offset_fp16 = j;
    int w_offset_uint8 = w_offset_fp16 / 2;
    int z_offset_uint8 = w_offset_fp16 / gs / 2;

    // loading quantized weights and zeros
    uint8_t qw[16];

    // uint8 -> half
    // half w[8], z[8];
    half w[32];
    uint16_t w_[32];
    float w_f[32];

    *(float4*)(&qw[0]) = *(float4*)(&qweight[row * qc_in + w_offset_uint8]);

    uint32_t const& i4s = reinterpret_cast<uint32_t const&>(qw[0]);
    const uint32_t top_i4s = i4s >> 8;

    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h1[0])
        : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h1[1])
        : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h1[2])
        : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h1[3])
        : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    h1[0] <<= 4;
    h1[2] <<= 4;

    uint32_t const& i4s1 = reinterpret_cast<uint32_t const&>(qw[4]);
    const uint32_t top_i4s1 = i4s1 >> 8;
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h2[0])
        : "r"(i4s1), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h2[1])
        : "r"(i4s1), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h2[2])
        : "r"(top_i4s1), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h2[3])
        : "r"(top_i4s1), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    h2[0] <<= 4;
    h2[2] <<= 4;

    uint32_t const& i4s2 = reinterpret_cast<uint32_t const&>(qw[8]);
    const uint32_t top_i4s2 = i4s2 >> 8;
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h3[0])
        : "r"(i4s2), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h3[1])
        : "r"(i4s2), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h3[2])
        : "r"(top_i4s2), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h3[3])
        : "r"(top_i4s2), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    h3[0] <<= 4;
    h3[2] <<= 4;

    uint32_t const& i4s3 = reinterpret_cast<uint32_t const&>(qw[12]);
    const uint32_t top_i4s3 = i4s3 >> 8;
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h4[0])
        : "r"(i4s3), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h4[1])
        : "r"(i4s3), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h4[2])
        : "r"(top_i4s3), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h4[3])
        : "r"(top_i4s3), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    h4[0] <<= 4;
    h4[2] <<= 4;

    // pack zeros
    int inter_zero_idx = w_offset_fp16 / gs % 2;
    // half z_r[2] = {z[inter_zero_idx], z[inter_zero_idx]};
    half z = zeros[row * c_in / gs + j / gs];
    // half z_r[2] = {z, z};

    // load scales in fp16
    half s = scales[row * c_in / gs + j / gs];
// half s_r[2] = {s, s};

// scaling
#pragma unroll
    for (int i = 0; i < 32; i++) {
      // *(half2*)(&w[i]) = __hsub2(*(half2*)(&w[i]), *(half2*)(&z_r[0]));
      w_f[i] = __half2float(w[i]) - __half2float(z);
    }

    // half2 temp_temp_sum = __hmul2(vec_val[0], *(half2*)(&w[0]));
    // #pragma unroll
    // for (int t = 1; t < 16; t++){
    //   temp_temp_sum = __hadd2(temp_temp_sum, __hmul2(vec_val[t],
    //   *(half2*)(&w[(t << 1)])));
    // }
    // temp_sum = __hadd2(temp_sum, __hmul2(temp_temp_sum, *(half2*)(&s_r[0])));
    float temp_sum = 0.0f;

#pragma unroll
    for (int t = 0; t < 16; t++) {
      temp_sum += w_f[(t << 1)] * __half2float(vec_val[t].x);
      temp_sum += w_f[(t << 1) + 1] * __half2float(vec_val[t].y);
    }

    sum = sum + temp_sum * __half2float(s);
  }

  // sum = __hadd(temp_sum.x, temp_sum.y);
  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tidx == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0f;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tidx == 0) {
    res[row] = __float2half(sum);
  }
}
