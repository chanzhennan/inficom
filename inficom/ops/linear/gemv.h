#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor gemv_acc_fp16(at::Tensor X, at::Tensor W);
at::Tensor gemv_acc_fp32(at::Tensor X, at::Tensor W);
at::Tensor gemv_acc_fp16_overlap(at::Tensor X, at::Tensor W);
at::Tensor gemv_acc_fp16_overlap_turbo(at::Tensor X, at::Tensor W);
void dequant_gemv_4bit_wo_bias(torch::Tensor output, torch::Tensor input,
                               torch::Tensor qweight, torch::Tensor zeros,
                               torch::Tensor scales, const int in_channel,
                               const int bpoc, const int out_channel,
                               const int group_size);

void dequant_gemv_4bit_wo_bias_lop3(torch::Tensor output, torch::Tensor input,
                                    torch::Tensor qweight, torch::Tensor zeros,
                                    torch::Tensor scales, const int in_channel,
                                    const int bpoc, const int out_channel,
                                    const int group_size);
