/*
    Utility functions.
*/
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define DIV_UP(x, y) ((x) + (y)-1) / (y)
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// LDG.128, LDS.128
#define FETCH_FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FETCH_FLOAT3(value) (reinterpret_cast<float3 *>(&(value))[0])
#define FETCH_FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])

__device__ __forceinline__ float4 _4bits_dequant_lop3(uint8_t *input) {
  uint32_t const &i4s = reinterpret_cast<uint32_t const &>(input);
  const uint32_t top_i4s = i4s >> 8;
  uint32_t h[4];

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOT_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t MAGIC_NUM_0 = 0x64006400;        // `1024`
  static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;        // `64`
  static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4;  // `64` >> 4

  asm("lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(h[0])
      : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
  asm("lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(h[1])
      : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
  asm("lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(h[2])
      : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
  asm("lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(h[3])
      : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
  h[0] <<= 4;
  h[2] <<= 4;
  // we don't need to subtract the magic nums because zeros will go through the
  // same dequant function and carry the same magic constant, the magic num will
  // be canceled out after subtracting zeros

  return *((float4 *)&h[0]);
}

__device__ __forceinline__ float4 _4bits_dequant(uint8_t *input) {
  half output[8];
  output[0] = static_cast<half>(input[0] >> 4);
  output[1] = static_cast<half>(input[0] & 0x0f);
  output[2] = static_cast<half>(input[1] >> 4);
  output[3] = static_cast<half>(input[1] & 0x0f);
  output[4] = static_cast<half>(input[2] >> 4);
  output[5] = static_cast<half>(input[2] & 0x0f);
  output[6] = static_cast<half>(input[3] >> 4);
  output[7] = static_cast<half>(input[3] & 0x0f);

  return *((float4 *)&output[0]);
}

__device__ __forceinline__ float warpReduceSum(float sum_val,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum_val +=
        __shfl_down_sync(0xffffffff, sum_val, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum_val +=
        __shfl_down_sync(0xffffffff, sum_val, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum_val +=
        __shfl_down_sync(0xffffffff, sum_val, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 1);  // 0-1, 2-3, 4-5, etc.
  return sum_val;
}

__device__ __forceinline__ half warpReduceSum(half result,
                                              unsigned int threadNum) {
  if (threadNum >= 32)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result,
                                             16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result,
                                             8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result,
                                             4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result,
                                             2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result,
                                             1));  // 0-1, 2-3, 4-5, etc.
  return result;
}

__device__ __forceinline__ half2 warpReduceSum(half2 result,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result,
                                              16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result,
                                              8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result,
                                              4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result,
                                              2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result,
                                              1));  // 0-1, 2-3, 4-5, etc.
  return result;
}

__device__ __forceinline__ float warpReduceMax(float max_val,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val,
                                            16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val,
                                            8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val,
                                            4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val,
                                            2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val,
                                            1));  // 0-1, 2-3, 4-5, etc.
  return max_val;
}

__device__ __forceinline__ float blockReduceSum(float reducing,
                                                float *shared_mem) {
  // Helper function for reduce softmax exp sum.
  const int32_t WPT = blockDim.x / 32;
  int32_t WPTB = WPT == 20 ? 32 : WPT;
  const int32_t lane_id = threadIdx.x % 32;
  const int32_t warp_id = threadIdx.x / 32;

#pragma unroll
  for (int32_t mask = 16; mask >= 1; mask /= 2) {
    reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
  }

  if (lane_id == 0) shared_mem[warp_id] = reducing;
  __syncthreads();

  if (lane_id < WPTB) reducing = lane_id < WPT ? shared_mem[lane_id] : 0.0f;

#pragma unroll
  for (int32_t mask = WPTB / 2; mask >= 1; mask /= 2) {
    reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
  }
  reducing = __shfl_sync(uint32_t(-1), reducing, 0);
  return reducing;
}

__device__ __forceinline__ half blockReduceSum(half reducing,
                                               half *shared_mem) {
  // Helper function for reduce softmax exp sum.
  const int32_t WPT = blockDim.x / 32;
  const int32_t lane_id = threadIdx.x % 32;
  const int32_t warp_id = threadIdx.x / 32;

#pragma unroll
  for (int32_t mask = 16; mask >= 1; mask /= 2) {
    reducing = __hadd(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
  }

  if (lane_id == 0) shared_mem[warp_id] = reducing;
  __syncthreads();

  if (lane_id < WPT) reducing = shared_mem[lane_id];

#pragma unroll
  for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
    reducing = __hadd(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
  }
  reducing = __shfl_sync(uint32_t(-1), reducing, 0);
  return reducing;
}
