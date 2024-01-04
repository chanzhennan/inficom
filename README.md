# 算子库

## 量化矩阵乘

### W4A16

1. 相关代码
    * 算子 kernel 在 `inficom/kernels/gemm_s4_f16`
    * 传入的权重需要重排，用法参考 `inficom/kernels/gemm_s4_f16/README.md`
    * 对应的 attn 和 FFN  量化层实现在 `inficom/layers` 下带 quant 的文件中
        - 通过 `LlamaAttnQuant`, `LlamaFFNQuant` 分别实例化对应量化层
        - 调用类中的 `Forward` 成员函数进行推理

2. 测试
    * 单算子 `python script/ops/w4_a16_test.py`
    * 量化层 `python script/layers/llama_attn_quant_test.py --n_bit 4`
