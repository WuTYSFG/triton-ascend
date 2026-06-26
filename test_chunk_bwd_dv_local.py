# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.common.chunk_o import chunk_bwd_dv_local
from fla.utils import assert_close, device


def naive_chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K = q.shape
    V = do.shape[-1]
    HV = do.shape[2]
    orig_dtype = q.dtype

    if scale is None:
        scale = K ** -0.5

    q = q.to(torch.float32)
    k = k.to(torch.float32)
    do = do.to(torch.float32)
    g = g.to(torch.float32)

    dv = torch.zeros(B, T, HV, V, dtype=torch.float32, device=q.device)

    NT = (T + chunk_size - 1) // chunk_size
    for i_b in range(B):
        for i_t in range(NT):
            s = i_t * chunk_size
            e = min(s + chunk_size, T)
            bt = e - s

            for i_h in range(HV):
                q_h = q[i_b, s:e, i_h // (HV // H), :]
                k_h = k[i_b, s:e, i_h // (HV // H), :]
                do_h = do[i_b, s:e, i_h, :]
                g_h = g[i_b, s:e, i_h]

                b_A = torch.matmul(k_h, q_h.transpose(-1, -2)) * scale
                b_A *= torch.exp(g_h[None, :] - g_h[:, None])

                mask = torch.triu(torch.ones(bt, bt, device=q.device, dtype=torch.float32))
                b_A = b_A * mask

                dv_chunk = torch.matmul(b_A, do_h)
                dv[i_b, s:e, i_h, :] = dv_chunk

    return dv.to(orig_dtype)


CASES = [
    (1, 1024, 32, 128),
    # (4, 1024, 32, 128),
    # (16, 1024, 32, 128),
    # (1, 8192, 32, 128),
    # (4, 8192, 32, 128),
    # (16, 8192, 8, 128),
    # (1, 131072, 4, 128),
    # (4, 131072, 2, 128),
    # (16, 131072, 2, 128),
    # (1, 16384, 4, 128),
]


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K'),
    [pytest.param(*test, id='B{}-T{}-H{}-K{}'.format(*test)) for test in CASES],
)
def test_chunk_bwd_dv_local_g(
    B: int,
    T: int,
    H: int,
    K: int,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", "1")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    torch.manual_seed(42)

    V = K
    dtype = torch.bfloat16
    chunk_size = 64
    HV = H

    q = torch.randn((B, T, H, K), dtype=dtype, device=device)
    k = torch.randn((B, T, H, K), dtype=dtype, device=device)
    do = torch.randn((B, T, HV, V), dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn((B, T, HV), dtype=dtype, device=device))

    scale = K ** -0.5

    ref_dv = naive_chunk_bwd_dv_local(
        q=q, k=k, do=do, g=g, scale=scale, chunk_size=chunk_size,
    )
    tri_dv = chunk_bwd_dv_local(
        q=q, k=k, do=do, g=g, scale=scale, chunk_size=chunk_size,
    )

    assert_close('dv', ref_dv, tri_dv, 0.005)
