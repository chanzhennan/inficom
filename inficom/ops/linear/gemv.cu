#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../../kernels/fast_gemv.cuh"

at::Tensor gemv_acc_fp16(at::Tensor X, at::Tensor W) {
  // X: [bs, 1, dim_in]
  // W: [dim_out, dim_in]

  if (X.size(2) != W.size(1)) {
    throw std::invalid_argument("embbed dim mismatch!");
  }
  if (X.size(2) % 128 != 0) {
    throw std::invalid_argument("embbed dim must be a multiple of 128!");
  }

  int bs = X.size(0);
  int dim_in = X.size(2);
  int dim_out = W.size(0);

  at::Tensor O = torch::empty(
      {bs, 1, dim_out}, at::device(X.device()).dtype(at::ScalarType::Half));

  fast_gemv_acc_fp16_kernel<<<dim3(1, dim_out / 2), dim3(128, 1)>>>(
      reinterpret_cast<half *>(W.data_ptr<at::Half>()),
      reinterpret_cast<half *>(X.data_ptr<at::Half>()),
      reinterpret_cast<half *>(O.data_ptr<at::Half>()), dim_in,
      DIV_UP(dim_in, 128));

  return O;
}

at::Tensor gemv_acc_fp32(at::Tensor X, at::Tensor W) {
  // X: [bs, 1, dim_in]
  // W: [dim_out, dim_in]

  if (X.size(2) != W.size(1)) {
    throw std::invalid_argument("embbed dim mismatch!");
  }
  if (X.size(2) % 128 != 0) {
    throw std::invalid_argument("embbed dim must be a multiple of 128!");
  }

  int bs = X.size(0);
  int dim_in = X.size(2);
  int dim_out = W.size(0);

  at::Tensor O = torch::empty(
      {bs, 1, dim_out}, at::device(X.device()).dtype(at::ScalarType::Half));

  fast_gemv_acc_fp32_kernel<<<dim3(1, dim_out), dim3(128, 1)>>>(
      reinterpret_cast<half *>(W.data_ptr<at::Half>()),
      reinterpret_cast<half *>(X.data_ptr<at::Half>()),
      reinterpret_cast<half *>(O.data_ptr<at::Half>()), dim_in,
      DIV_UP(dim_in, 128));

  return O;
}

at::Tensor gemv_acc_fp16_overlap(at::Tensor X, at::Tensor W) {
  // X: [bs, 1, dim_in]
  // W: [dim_out, dim_in]

  if (X.size(2) != W.size(1)) {
    throw std::invalid_argument("embbed dim mismatch!");
  }
  if (X.size(2) % 128 != 0) {
    throw std::invalid_argument("embbed dim must be a multiple of 128!");
  }

  int bs = X.size(0);       // m
  int dim_in = X.size(2);   // k
  int dim_out = W.size(0);  // n

  at::Tensor O = torch::empty(
      {bs, 1, dim_out}, at::device(X.device()).dtype(at::ScalarType::Half));

  fast_gemv_acc_fp16_kernel_overlap<<<dim3(1, dim_out), dim3(128, 1)>>>(
      reinterpret_cast<half *>(W.data_ptr<at::Half>()),
      reinterpret_cast<half *>(X.data_ptr<at::Half>()),
      reinterpret_cast<half *>(O.data_ptr<at::Half>()), dim_in,
      DIV_UP(dim_in, 128));

  return O;
}

at::Tensor gemv_acc_fp16_overlap_turbo(at::Tensor X, at::Tensor W) {
  // X: [bs, 1, dim_in]
  // W: [dim_out, dim_in]

  if (X.size(2) != W.size(1)) {
    throw std::invalid_argument("embbed dim mismatch!");
  }
  if (X.size(2) % 128 != 0) {
    throw std::invalid_argument("embbed dim must be a multiple of 128!");
  }

  int bs = X.size(0);       // m
  int dim_in = X.size(2);   // k
  int dim_out = W.size(0);  // n

  at::Tensor O = torch::empty(
      {bs, 1, dim_out}, at::device(X.device()).dtype(at::ScalarType::Half));

  fast_gemv_acc_fp16_kernel_overlap_turbo<<<dim3(1, dim_out / 2),
                                            dim3(128, 1)>>>(
      reinterpret_cast<half *>(W.data_ptr<at::Half>()),
      reinterpret_cast<half *>(X.data_ptr<at::Half>()),
      reinterpret_cast<half *>(O.data_ptr<at::Half>()), dim_in,
      DIV_UP(dim_in, 128));

  return O;
}

void dequant_gemv_4bit_wo_bias(torch::Tensor output, torch::Tensor input,
                               torch::Tensor qweight, torch::Tensor zeros,
                               torch::Tensor scales, const int in_channel,
                               const int bpoc, const int out_channel,
                               const int group_size) {
  if (input.size(2) != in_channel) {
    throw std::invalid_argument("Input channel size mismatch!");
  }
  if (output.size(2) != out_channel) {
    throw std::invalid_argument("Output channel size mismatch!");
  }
  if (qweight.size(1) != bpoc) {
    throw std::invalid_argument("Quantized input channel size mismatch!");
  }
  if (input.size(0) != 1) {
    throw std::invalid_argument("Only batchsize = 1 is allowed!");
  }
  if (output.size(0) != 1) {
    throw std::invalid_argument("Only batchsize = 1 is allowed!");
  }
  if (input.size(1) != 1) {
    throw std::invalid_argument("Only sequence length = 1 is allowed!");
  }
  if (output.size(1) != 1) {
    throw std::invalid_argument("Only sequence length = 1 is allowed!");
  }
  uint8_t *qweight_ptr = qweight.data_ptr<uint8_t>();
  half *zeros_ptr = reinterpret_cast<half *>(zeros.data_ptr<at::Half>());
  half *scales_ptr = reinterpret_cast<half *>(scales.data_ptr<at::Half>());
  half *input_ptr = reinterpret_cast<half *>(input.data_ptr<at::Half>());
  half *output_ptr = reinterpret_cast<half *>(output.data_ptr<at::Half>());

  int mat_height_ = out_channel;
  int vec_height_ = in_channel;

  int block_dim_x = 128;
  int block_dim_y = 1;
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = vec_height_ / block_dim_x;
  assert(num_per_thread >= 8);

  dim3 grid_dim(1, mat_height_ / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);

  dequant_gemv_4bit_wo_bias<<<grid_dim, block_dim>>>(
      qweight_ptr, input_ptr, output_ptr, vec_height_, num_per_thread,
      zeros_ptr, scales_ptr, in_channel, bpoc, out_channel, group_size);
}

void dequant_gemv_4bit_wo_bias_lop3(torch::Tensor output, torch::Tensor input,
                                    torch::Tensor qweight, torch::Tensor zeros,
                                    torch::Tensor scales, const int in_channel,
                                    const int bpoc, const int out_channel,
                                    const int group_size) {
  if (input.size(2) != in_channel) {
    throw std::invalid_argument("Input channel size mismatch!");
  }
  if (output.size(2) != out_channel) {
    throw std::invalid_argument("Output channel size mismatch!");
  }
  if (qweight.size(1) != bpoc) {
    throw std::invalid_argument("Quantized input channel size mismatch!");
  }
  if (input.size(0) != 1) {
    throw std::invalid_argument("Only batchsize = 1 is allowed!");
  }
  if (output.size(0) != 1) {
    throw std::invalid_argument("Only batchsize = 1 is allowed!");
  }
  if (input.size(1) != 1) {
    throw std::invalid_argument("Only sequence length = 1 is allowed!");
  }
  if (output.size(1) != 1) {
    throw std::invalid_argument("Only sequence length = 1 is allowed!");
  }
  uint8_t *qweight_ptr = qweight.data_ptr<uint8_t>();
  half *zeros_ptr = reinterpret_cast<half *>(zeros.data_ptr<at::Half>());
  half *scales_ptr = reinterpret_cast<half *>(scales.data_ptr<at::Half>());
  half *input_ptr = reinterpret_cast<half *>(input.data_ptr<at::Half>());
  half *output_ptr = reinterpret_cast<half *>(output.data_ptr<at::Half>());

  int mat_height_ = out_channel;
  int vec_height_ = in_channel;

  int block_dim_x = 128;
  int block_dim_y = 1;
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = vec_height_ / block_dim_x;
  assert(num_per_thread >= 8);

  dim3 grid_dim(1, mat_height_ / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);

  dequant_gemv_4bit_wo_bias_lop3<<<grid_dim, block_dim>>>(
      qweight_ptr, input_ptr, output_ptr, vec_height_, num_per_thread,
      zeros_ptr, scales_ptr, in_channel, bpoc, out_channel, group_size);
}
