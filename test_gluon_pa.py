from typing import Tuple, List, Dict
import re
import torch
import pytest
import numpy as np
import hashlib

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy
from triton.experimental.gluon.language.extra import libdevice

from aiter.test_common import perftest

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
print(f"THREADS_PER_WARP={THREADS_PER_WARP}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(123)


@gluon.jit
def _paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk(
    exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,       # [num_seqs]
    scale,
    k_scale,
    v_scale,
    alibi_slopes,
    stride_max_logits_s,
    stride_max_logits_nh,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_nh,
    stride_logits_p,
    stride_logits_g,
    stride_q_s,
    stride_q_nh,
    stride_k_b,
    stride_k_nh,
    stride_k_hz,
    stride_k_bz,
    stride_v_b,
    stride_v_nh,
    stride_v_hz,
    stride_bt_s,
    compute_type: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_part_idx = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634
    CONTIGUOUS_KV_ELEMS_16B_LOAD: tl.constexpr = 8
    K_HEAD_SZ_POW2_SPLIT: tl.constexpr = HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    MAX_NUM_KV_BLKS: tl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    # 1 x QUERY_GRP_SZ_POW2 x HEAD_SZ_POW2
    # 1 x 8(mdim) x 128(kdim)
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[1, 1, 4],
        threads_per_warp=[1, 8, 8],
        warps_per_cta   =[1, 1, 4],
        order           =[2, 1, 0],
    )
    # MAX_NUM_KV_BLKS x K_HEAD_SZ_POW2_SPLIT x KV_BLK_SZ_POW2 x CONTIGUOUS_KV_ELEMS_16B_LOAD
    # 16 x 16 x 16 x 8
    blocked_k: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[4, 2, 2, 8],
        threads_per_warp=[1, 8, 8, 1],
        warps_per_cta   =[4, 1, 1, 1],
        order           =[3, 2, 1, 0],
    )
    # MAX_NUM_KV_BLKS x HEAD_SZ_POW2 x KV_BLK_SZ_POW2
    # 16 x 128(kdim) x 16(ndim)
    blocked_kt: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[4, 4, 8],
        threads_per_warp=[1, 32, 2],
        warps_per_cta   =[4, 1, 1],
        order           =[2, 1, 0],
    )

    query_grp_sz_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, blocked_q))
    head_sz: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, blocked_q))





    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_div_offs = tl.arange(0, K_HEAD_SZ_POW2_SPLIT)
    contiguous_kv_elems_offs = tl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    # load all kv blocks in one time
    blk_ids = tl.arange(0, MAX_NUM_KV_BLKS)
    masked_blk_ids = tl.where(blk_ids < num_kv_blks, blk_ids, 0)
    kv_blk_start = seq_part_idx * MAX_NUM_KV_BLKS
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_start + masked_blk_ids)

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(compute_type)

    # k_blk_offs[MAX_NUM_KV_BLKS, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_blk_offs = (
        kv_blk_nums[:, None, None, None] * stride_k_b
        + kv_head_idx * stride_k_nh
        + head_sz_div_offs[None, :, None, None] * stride_k_hz
        + blk_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
        + contiguous_kv_elems_offs[None, None, None, :]
    )
    # blk_seq_offs[MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2]
    blk_seq_offs = ((kv_blk_start + blk_ids[:, None]) * KV_BLK_SZ  # blk_ids: [MAX_NUM_KV_BLKS]
                    + blk_offs[None, :]) # blk_offs: [KV_BLK_SZ_POW2]

    k_mask = (
        (blk_seq_offs[:, None, :, None] < seq_len) &
        (blk_offs[None, None, :, None] < KV_BLK_SZ) &
        (head_sz_div_offs[None, :, None, None] < (HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD))
    )

    # k[MAX_NUM_KV_BLKS, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_0 = tl.load(k_cache_ptr + k_blk_offs)
    k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
    k = k.to(compute_type)
    # k[HEAD_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    k = tl.permute(k, [1, 3, 0, 2]) # [HEAD_SZ_POW2/x, x, MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2]
    k = tl.reshape(k, [HEAD_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])

    # qk[QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    qk = tl.dot(q, k, out_dtype=tl.float32)
    blk_seq_flatten_offs = tl.reshape(blk_seq_offs, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])
    if alibi_slopes is not None:
        qk += (alibi_slope[:, None] * (blk_seq_flatten_offs - seq_len + 1)[None, :]).to(
            tl.float32
        )
    qk = tl.where(
        (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_flatten_offs[None, :] < seq_len),
        qk,
        float("-inf"),
    )

    max_logit_new = tl.max(qk, axis=1)
    # p[QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
    p = p.to(compute_type)
    exp_sum = tl.sum(p, axis=1)

    # v_blk_offs[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    v_blk_offs = (
        kv_blk_nums[:, None, None] * stride_v_b
        + kv_head_idx * stride_v_nh
        + head_sz_offs[None, :, None] * stride_v_hz
        + blk_offs[None, None, :]
    )
    v_mask = (
        (blk_seq_offs[:, None, :] < seq_len) &
        (blk_offs[None, None, :] < KV_BLK_SZ) &
        (head_sz_offs[None, :, None] < HEAD_SZ)
    )

    # v[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    v_0 = tl.load(v_cache_ptr + v_blk_offs)
    v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
    v = v.to(compute_type)
    # v[MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2]
    v = tl.permute(v, [0, 2, 1])
    v = tl.reshape(v, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2])

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs, max_logit_new, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)

    # acc[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    acc = tl.dot(p, v, out_dtype=tl.float32)
    acc = acc / exp_sum[:, None]

    # end up computation
    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )
    tl.store(logits_ptr + logits_offs, acc, mask=q_mask)

