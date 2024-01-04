#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

// #include "kernels/gemm_s4_f16/format.h"
// #include "kernels/gemm_s4_f16/gemm_s4_f16.h"
// #include "layers/layer.h"
// #include "ops/attn/decode_attn.h"
// #include "ops/element/residual.h"
#include "ops/linear/gemv.h"
// #include "ops/linear/w8.h"
// #include "ops/norm/norm.h"

PYBIND11_MODULE(inficom, m) {
  // ops: decode attention
  // m.def("decode_mha_with_async_softmax", &decode_mha_with_async_softmax);
  // m.def("decode_mha_fall_back", &decode_mha_fall_back);
  // m.def("decode_mqa_with_async_softmax", &decode_mha_with_async_softmax);
  // m.def("decode_mqa_fall_back", &decode_mha_fall_back);
  // // ops: norm
  // m.def("rmsnorm", &rmsnorm);
  // m.def("layernorm", &layernorm);
  // // ops: add residual
  // m.def("add_residual", &add_residual);
  // // ops: fused add residual and norm
  // m.def("residual_rmsnorm", &residual_rmsnorm);
  // m.def("residual_layernorm", &residual_layernorm);
  // ops: linear - gemv
  m.def("gemv_acc_fp16", &gemv_acc_fp16);
  m.def("gemv_acc_fp32", &gemv_acc_fp32);
  m.def("gemv_acc_fp16_overlap", &gemv_acc_fp16_overlap);
  m.def("gemv_acc_fp16_overlap_turbo", &gemv_acc_fp16_overlap_turbo);
  m.def("dequant_gemv_4bit_wo_bias", &dequant_gemv_4bit_wo_bias);
  m.def("dequant_gemv_4bit_wo_bias_lop3", &dequant_gemv_4bit_wo_bias_lop3);

  // layers: attn
  //     m.def("llama2_attn_layer_fwd", &llama2_attn_layer_fwd);
  //     m.def("llama2_before_attn_layer_fwd", &llama2_before_attn_layer_fwd);
  //     m.def("llama2_decode_attn_layer_fwd", &llama2_decode_attn_layer_fwd);
  //     m.def("llama2_after_attn_layer_fwd", &llama2_after_attn_layer_fwd);
  //     m.def("chatglm2_attn_layer_fwd", &chatglm2_attn_layer_fwd);
  //     m.def("opt_attn_layer_fwd", &opt_attn_layer_fwd);
  //     // layers: ffn
  //     m.def("llama2_ffn_layer_fwd", &llama2_ffn_layer_fwd);
  //     m.def("chatglm2_ffn_layer_fwd", &chatglm2_ffn_layer_fwd);
  //     m.def("opt_ffn_layer_fwd", &opt_ffn_layer_fwd);

  //     // quant layers: attn
  //     pybind11::class_<LlamaAttnQuant>(m, "LlamaAttnQuant")
  //         .def(pybind11::init<>())
  //         .def("Forward", &LlamaAttnQuant::Forward);
  //     // quant layers: ffn
  //     pybind11::class_<LlamaFFNQuant>(m, "LlamaFFNQuant")
  //         .def(pybind11::init<>())
  //         .def("Forward", &LlamaFFNQuant::Forward);

  //     // W4A16 lmdeploy
  // m.def("convert_ours_to_awq", &convert_ours_to_awq, "convert qweight from
  // ours to awq."); m.def("convert_awq_to_lmdeploy", &convert_awq_to_lmdeploy,
  // "convert qweight from awq to lmdeploy.");
  // m.def("transpose_merge_zeros_scales", &transpose_merge_zeros_scales,
  // "reformat: zeros & scales.");

  //     pybind11::class_<turbomind::GemmS4F16>(m, "GemmS4F16")
  //         .def(pybind11::init<>())
  //         .def("RunWrapper", &turbomind::GemmS4F16::RunWrapper)
  //         .def("Run", &turbomind::GemmS4F16::Run);
  //     // end W4A16 lmdeploy

  //     // W8A16
  //     m.def("i16_w8_c16_o16_wo_bias", &i16_w8_c16_o16_wo_bias);
  //     m.def("llama2_attn_w8_layer_fwd", &llama2_attn_w8_layer_fwd);
  //     m.def("llama2_ffn_w8_layer_fwd", &llama2_ffn_w8_layer_fwd);
}
