import torch
import torch.nn as nn
from inficom import (
    gemv_acc_fp16,
    gemv_acc_fp32,
    gemv_acc_fp16_overlap,
    gemv_acc_fp16_overlap_turbo,
    dequant_gemv_4bit_wo_bias,
    dequant_gemv_4bit_wo_bias_lop3,
)
from quant_dequant import generate_quant

### test settings (Z must be 1 for GEMV testing)
Z = 1
DIM1 = 11008
DIM2 = 4096

### benchmark settings
WARM_UP = 25
REP = 100

x = torch.empty((Z, 1, DIM1), dtype=torch.float16, device="cuda").normal_(
    mean=0.0, std=0.5
)
linear_layer = torch.nn.Linear(
    DIM1, DIM2, bias=False, device="cuda", dtype=torch.float16
)

ref_out = linear_layer(x)

### quant
quant_bit = 4
group_size = 128
qweight, zeros, scales = generate_quant(
    linear_layer.weight, DIM1, quant_bit, group_size
)


#### quant


ed1_out = gemv_acc_fp16(x, linear_layer.weight)
ed2_out = gemv_acc_fp32(x, linear_layer.weight)
ed3_out = gemv_acc_fp16_overlap(x, linear_layer.weight)
ed4_out = gemv_acc_fp16_overlap_turbo(x, linear_layer.weight)

ed5_out = torch.empty(
    ed4_out.shape, dtype=torch.float16, device="cuda"
).normal_(mean=0.0, std=0.5)
dequant_gemv_4bit_wo_bias(
    ed5_out, x, qweight, zeros, scales, DIM1, DIM1 // 2, DIM2, 128
)

ed6_out = torch.empty(
    ed4_out.shape, dtype=torch.float16, device="cuda"
).normal_(mean=0.0, std=0.5)
dequant_gemv_4bit_wo_bias_lop3(
    ed6_out, x, qweight, zeros, scales, DIM1, DIM1 // 2, DIM2, 128
)


ed1_all_close = torch.allclose(
    ref_out, ed1_out.reshape((Z, 1, DIM2)), atol=1e-2, rtol=1e-4
)
ed2_all_close = torch.allclose(
    ref_out, ed2_out.reshape((Z, 1, DIM2)), atol=1e-2, rtol=1e-4
)
ed3_all_close = torch.allclose(
    ref_out, ed3_out.reshape((Z, 1, DIM2)), atol=1e-2, rtol=1e-4
)
ed4_all_close = torch.allclose(
    ref_out, ed4_out.reshape((Z, 1, DIM2)), atol=1e-2, rtol=1e-4
)
ed5_all_close = torch.allclose(
    ref_out, ed5_out.reshape((Z, 1, DIM2)), atol=1e-1, rtol=1e-4
)
ed6_all_close = torch.allclose(
    ref_out, ed6_out.reshape((Z, 1, DIM2)), atol=1e-1, rtol=1e-4
)

##############################


### benchmarking
for _ in range(WARM_UP):
    _ = linear_layer(x)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = linear_layer(x)
    end_event[i].record()
torch.cuda.synchronize()
ref_dur = torch.tensor(
    [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
    dtype=torch.float,
)

##############################


for _ in range(WARM_UP):
    _ = gemv_acc_fp16(x, linear_layer.weight)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = gemv_acc_fp16(x, linear_layer.weight)
    end_event[i].record()
torch.cuda.synchronize()
ed1_dur = torch.tensor(
    [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
    dtype=torch.float,
)

##############################


for _ in range(WARM_UP):
    _ = gemv_acc_fp32(x, linear_layer.weight)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = gemv_acc_fp32(x, linear_layer.weight)
    end_event[i].record()
torch.cuda.synchronize()
ed2_dur = torch.tensor(
    [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
    dtype=torch.float,
)


##############################


for _ in range(WARM_UP):
    _ = gemv_acc_fp16_overlap(x, linear_layer.weight)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = gemv_acc_fp16_overlap(x, linear_layer.weight)
    end_event[i].record()
torch.cuda.synchronize()
ed3_dur = torch.tensor(
    [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
    dtype=torch.float,
)


##############################


for _ in range(WARM_UP):
    _ = gemv_acc_fp16_overlap_turbo(x, linear_layer.weight)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = gemv_acc_fp16_overlap_turbo(x, linear_layer.weight)
    end_event[i].record()
torch.cuda.synchronize()
ed4_dur = torch.tensor(
    [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
    dtype=torch.float,
)


############################## quant

for _ in range(WARM_UP):
    dequant_gemv_4bit_wo_bias(
        ed5_out, x, qweight, zeros, scales, DIM1, DIM1 // 2, DIM2, 128
    )


start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    dequant_gemv_4bit_wo_bias(
        ed5_out, x, qweight, zeros, scales, DIM1, DIM1 // 2, DIM2, 128
    )
    end_event[i].record()
torch.cuda.synchronize()
ed5_dur = torch.tensor(
    [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
    dtype=torch.float,
)


############################## quant

for _ in range(WARM_UP):
    dequant_gemv_4bit_wo_bias_lop3(
        ed6_out, x, qweight, zeros, scales, DIM1, DIM1 // 2, DIM2, 128
    )


start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    dequant_gemv_4bit_wo_bias_lop3(
        ed6_out, x, qweight, zeros, scales, DIM1, DIM1 // 2, DIM2, 128
    )
    end_event[i].record()
torch.cuda.synchronize()
ed6_dur = torch.tensor(
    [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
    dtype=torch.float,
)


print(
    '%s %s %s %s %s %s %d %d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
    % (
        bool(ed1_all_close),
        bool(ed2_all_close),
        bool(ed3_all_close),
        bool(ed4_all_close),
        bool(ed5_all_close),
        bool(ed6_all_close),
        Z,
        DIM1,
        DIM2,
        torch.mean(ref_dur).item(),
        torch.mean(ed1_dur).item(),
        torch.mean(ed2_dur).item(),
        torch.mean(ed3_dur).item(),
        torch.mean(ed4_dur).item(),
        torch.mean(ed5_dur).item(),
        torch.mean(ed6_dur).item(),
    )
)
