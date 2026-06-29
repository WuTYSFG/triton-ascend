"""
最小复现：和原始 bug 完全相同结构的三段 cat + 间接索引 + int64
保持 compile_mode="unstructured_in_simt" 触发同样的 SIMT 路径
"""
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_minimal(
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5,
    in_ptr6, in_ptr7, out_ptr0,
    y0_numel, x1_numel,
    Y0BLOCK: tl.constexpr, Y0BLOCK_SUB: tl.constexpr, X1BLOCK_SUB: tl.constexpr,
):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    tl.device_print("y0_offset: ", y0_offset)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
        y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :]
            x1_mask = x1 < x1_numel

            # Block 2: load int64 index from in_ptr2 at offset (-64+y0)
            tmp27 = (y0 >= 64) & (y0 < 128) & y0_mask
            tmp31 = tl.load(in_ptr2 + ((-64) + y0), tmp27, other=0)

            # Block 3: load int64 index from in_ptr2 at offset (-128+y0)
            tmp48 = (y0 >= 128) & y0_mask
            tmp52 = tl.load(in_ptr2 + ((-128) + y0), tmp48, other=0)

            # Use indices to load from in_ptr3
            tmp19 = tl.load(in_ptr3 + (x1 + 2048 * tmp31), x1_mask, other=0.0)
            tmp41 = tl.load(in_ptr3 + (x1 + 2048 * tmp52), x1_mask, other=0.0)

            result = tl.where(tmp48, tmp41, tl.where(tmp27, tmp19, 0.0))
            tl.store(out_ptr0 + (x1 + 2048 * y0), result, x1_mask & y0_mask)


def test():
    device = 'npu:0'
    arg0_1 = torch.randint(0, 95, (1, 64), device=device, dtype=torch.int64)  # in_ptr2
    arg10_1 = torch.randn((96, 2048), device=device, dtype=torch.float32)     # in_ptr3
    buf0 = torch.randn((64, 2048), device=device, dtype=torch.float32)         # in_ptr0
    buf1 = torch.randn((64, 2048), device=device, dtype=torch.float32)         # in_ptr4
    buf2 = torch.randn((64, 2048), device=device, dtype=torch.float32)         # in_ptr6
    arg3_1 = torch.randn((2048,), device=device, dtype=torch.float32)           # in_ptr5
    arg6_1 = torch.randn((2048,), device=device, dtype=torch.float32)           # in_ptr7
    arg9_1 = torch.randn((2048,), device=device, dtype=torch.float32)           # in_ptr1
    buf3 = torch.empty((1, 192, 2048), device=device, dtype=torch.float32)      # out_ptr0
    kernel_minimal[(36,)](
        buf0, arg9_1, arg0_1, arg10_1, buf1, arg3_1, buf2, arg6_1, buf3,
        192, 2048, 6, 4, 2048,
        compile_mode="unstructured_in_simt", multibuffer=False,
    )
    # ===== 加在这里 =====
    val = buf0[0, :6].float().cpu()
    print(f"B2: UBoff={val[0]:.0f} size={val[1]:.0f} GMoff={val[2]:.0f}")
    print(f"B3: UBoff={val[3]:.0f} size={val[4]:.0f} GMoff={val[5]:.0f}")
    # ===================
    torch.npu.synchronize()
    print("PASS")


if __name__ == "__main__":
    test()
