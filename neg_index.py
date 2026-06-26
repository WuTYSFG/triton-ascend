# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
import triton.language.extra.cann.extension as tl_math
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._C import _npu_getCurrentRawStreamNoWait as get_raw_stream
import torch_npu
torch_npu.npu._initialized = torch_npu.npu.is_initialized()
aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p

# import pydevd_pycharm
# pydevd_pycharm.settrace('141.5.149.114', port=8889)
# kernel path: /tmp/torchinductor_root/nc/cncgg6qaecgkb5xgzizmkhlo4nfpxbyxyaqnvgcdcf6cz5n6oskd.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2_replacement, %add_replacement, %add_1_replacement], 1), kwargs = {})
# SchedulerNodes: [SchedulerNode(name='op3')]

triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
if not torch_npu.npu.is_initialized():
    torch_npu.npu._initialized = True
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints={'y0': 192, 'x1': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'backend_hash': '470ECFBB241035716FB73877EA9F3E4C093EC10617F2DE9C5B6BBA6AE4BB1753', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 192), ('x1', 2048)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'group_enabled': False, 'group_template': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'runtime_block_arg_names': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, y0_numel, x1_numel, Y0BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
            x1_mask = x1 < x1_numel
            tmp7 = tl.load(in_ptr1 + (x1), x1_mask)
            tmp29 = tl.load(in_ptr5 + (x1), x1_mask)
            tmp50 = tl.load(in_ptr7 + (x1), x1_mask)
            tmp0 = y0
            tmp1 = tl.full([1, 1], 0, tl.int64)
            tmp2 = tmp0 >= tmp1
            tmp3 = tl.full([1, 1], 64, tl.int64)
            tmp4 = tmp0 < tmp3
            tmp5 = tmp4 & y0_mask
            tmp6 = tl.load(in_ptr0 + (x1 + 2048*(y0)), x1_mask & tmp5, other=0.0)
            tmp8 = tmp6 + tmp7
            tmp9 = tl.load(in_ptr2 + (y0), tmp5, other=0.0)
            tmp10 = tl.full([1, 1], 0, tl.int64)
            tmp11 = tl.maximum(tmp9, tmp10, tl.PropagateNan.ALL)
            tmp12 = tl.full([1, 1], 95, tl.int64)
            tmp13 = tl.minimum(tmp11, tmp12, tl.PropagateNan.ALL)
            tmp14 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 96, tl.int32)
            tmp15 = tmp13 + tmp14
            tmp16 = tmp13 < 0
            tmp17 = tl.where(tmp16, tmp15, tmp13)
            tl.device_assert((0 <= tmp17) & (tmp17 < 96), "index out of bounds: 0 <= tmp17 < 96")
            tmp19 = tl.load(in_ptr3 + (x1 + 2048*tmp17), x1_mask)
            tmp20 = tmp8 + tmp19
            tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
            tmp22 = tl.where(tmp5, tmp20, tmp21)
            tmp23 = tmp0 >= tmp3
            tmp24 = tl.full([1, 1], 128, tl.int64)
            tmp25 = tmp0 < tmp24
            tmp26 = tmp23 & tmp25
            tmp27 = tmp26 & y0_mask
            tmp28 = tl.load(in_ptr4 + (x1 + 2048*((-64) + y0)), x1_mask & tmp27, other=0.0)
            tmp30 = tmp28 + tmp29
            tmp31 = tl.load(in_ptr2 + ((-64) + y0), tmp27, other=0.0)
            
            tmp32 = tl.full([1, 1], 0, tl.int64)
            tmp33 = tl.maximum(tmp31, tmp32, tl.PropagateNan.ALL)
            tmp34 = tl.full([1, 1], 95, tl.int64)
            tmp35 = tl.minimum(tmp33, tmp34, tl.PropagateNan.ALL)
            tmp36 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 96, tl.int32)
            tmp37 = tmp35 + tmp36
            tmp38 = tmp35 < 0
            tmp39 = tl.where(tmp38, tmp37, tmp35)
            tl.device_assert((0 <= tmp39) & (tmp39 < 96), "index out of bounds: 0 <= tmp39 < 96")
            tmp41 = tl.load(in_ptr3 + (x1 + 2048*tmp39), x1_mask)
            tmp42 = tmp30 + tmp41
            tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
            tmp44 = tl.where(tmp27, tmp42, tmp43)
            tmp45 = tmp0 >= tmp24
            tmp46 = tl.full([1, 1], 192, tl.int64)
            tmp47 = tmp0 < tmp46
            tmp48 = tmp45 & y0_mask
            tmp49 = tl.load(in_ptr6 + (x1 + 2048*((-128) + y0)), x1_mask & tmp48, other=0.0)
            tmp51 = tmp49 + tmp50
            tmp52 = tl.load(in_ptr2 + ((-128) + y0), tmp48, other=0.0)
            tmp53 = tl.full([1, 1], 0, tl.int64)
            tmp54 = tl.maximum(tmp52, tmp53, tl.PropagateNan.ALL)
            tmp55 = tl.full([1, 1], 95, tl.int64)
            tmp56 = tl.minimum(tmp54, tmp55, tl.PropagateNan.ALL)
            tmp57 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 96, tl.int32)
            tmp58 = tmp56 + tmp57
            tmp59 = tmp56 < 0
            tmp60 = tl.where(tmp59, tmp58, tmp56)
            tl.device_assert((0 <= tmp60) & (tmp60 < 96), "index out of bounds: 0 <= tmp60 < 96")
            tmp62 = tl.load(in_ptr3 + (x1 + 2048*tmp60), x1_mask)
            tmp63 = tmp51 + tmp62
            tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
            tmp65 = tl.where(tmp48, tmp63, tmp64)
            tmp66 = tl.where(tmp26, tmp44, tmp65)
            tmp67 = tl.where(tmp4, tmp22, tmp66)
            tl.store(out_ptr0 + (x1 + 2048*y0), tmp67, x1_mask & y0_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

def call():
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        from torch._dynamo.testing import rand_strided
        from torch._inductor.utils import print_performance
        arg0_1 = rand_strided((1, 64), (64, 1), device='npu:0', dtype=torch.int64)
        arg3_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
        arg6_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
        arg9_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
        arg10_1 = rand_strided((96, 2048), (2048, 1), device='npu:0', dtype=torch.float32)
        buf0 = rand_strided((64, 2048), (2048, 1), device='npu', dtype=torch.float32)
        buf1 = rand_strided((64, 2048), (2048, 1), device='npu', dtype=torch.float32)
        buf2 = rand_strided((64, 2048), (2048, 1), device='npu', dtype=torch.float32)
        buf3 = empty_strided((1, 192, 2048), (393216, 2048, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(buf0, arg9_1, arg0_1, arg10_1, buf1, arg3_1, buf2, arg6_1, buf3, 192, 2048, stream=stream0)
        # triton_poi_fused_cat_0[(36,)](buf0, arg9_1, arg0_1, arg10_1, buf1, arg3_1, buf2, arg6_1, buf3, 192, 2048, 6, 4, 2048, compile_mode="unstructured_in_simt", multibuffer=False)
        del arg0_1
        del arg10_1
        del arg3_1
        del arg6_1
        del arg9_1
        del buf0
        del buf1
        del buf2
    return (buf3, )

call()
