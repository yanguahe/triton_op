from typing import Tuple, List, Dict
from typing import Optional
import re
import torch
import pytest
import numpy as np
import hashlib
import math
import tempfile
import os

import triton
import triton.language as tl
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import compile

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy
from triton.experimental.gluon.language.extra import libdevice

from aiter.test_common import perftest


TEST_NUM_ITERS = 101

# This code is derived from sglang and FLASHNN projects
# https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py

_SEQ_PARTITION_SIZE = 256  # HIP
ttgir_file_path = os.path.join(os.path.dirname(__file__), "./ttgir/pa_noloop.ttgir")

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
print(f"THREADS_PER_WARP={THREADS_PER_WARP}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(123)


@gluon.jit
def _paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk_gluon(
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
    compute_type: gl.constexpr,
    HEAD_SZ: gl.constexpr,
    HEAD_SZ_POW2: gl.constexpr,
    QUERY_GRP_SZ: gl.constexpr,
    QUERY_GRP_SZ_POW2: gl.constexpr,
    KV_BLK_SZ: gl.constexpr,
    KV_BLK_SZ_POW2: gl.constexpr,
    SEQ_PARTITION_SZ: gl.constexpr,
):
    """
    #TODO: Add Doc
    """

    seq_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    seq_part_idx = gl.program_id(2)

    log2e: gl.constexpr = 1.4426950408889634
    CONTIGUOUS_KV_ELEMS_16B_LOAD: gl.constexpr = 8
    K_HEAD_SZ_POW2_SPLIT: gl.constexpr = HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD

    seq_len = gl.load(seq_lens_ptr + seq_idx)
    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = gl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    MAX_NUM_KV_BLKS: gl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    num_kv_blks = gl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    # # 1 x QUERY_GRP_SZ_POW2 x HEAD_SZ_POW2
    # # 1 x 8(mdim) x 128(kdim)
    # blocked_q: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread =[1, 1, 4],
    #     threads_per_warp=[1, 8, 8],
    #     warps_per_cta   =[1, 1, 4],
    #     order           =[2, 1, 0],
    # )
    # QUERY_GRP_SZ_POW2 x HEAD_SZ_POW2
    # 16(mdim) x 128(kdim)
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[2, 4],
        threads_per_warp=[8, 8],
        warps_per_cta   =[1, 4],
        order           =[1, 0],
    )
    # MAX_NUM_KV_BLKS x K_HEAD_SZ_POW2_SPLIT x KV_BLK_SZ_POW2 x CONTIGUOUS_KV_ELEMS_16B_LOAD
    # 16 x 16 x 16 x 8
    blocked_k: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[4, 2, 2, 8],
        threads_per_warp=[1, 8, 8, 1],
        warps_per_cta   =[4, 1, 1, 1],
        order           =[3, 2, 1, 0],
    )

    # transposed: indicates the result tensor is transposed so that each thread holds consecutive elements
    # in the same row instead of column, which is good for chained dot and global write.
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        # version=3, instr_shape=[16, 16], transposed=False, warps_per_cta=[4, 1]
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    qk_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=16
    )
    qk_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=16
    )

    pv_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=False, warps_per_cta=[1, 4]
    )
    pv_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_mfma_layout, k_width=4
    )
    pv_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=4
    )

    # # blocked_q dim1
    # query_grp_sz_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, blocked_q))
    # # blocked_q dim2
    # head_sz_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, blocked_q))
    # blocked_q dim0
    query_grp_sz_layout: gl.constexpr = gl.SliceLayout(1, blocked_q)
    # blocked_q dim1
    head_sz_layout: gl.constexpr = gl.SliceLayout(0, blocked_q)

    # blocked_k dim0
    blk_id_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_k)))
    # blocked_k dim1
    head_sz_div_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_k)))
    # blocked_k dim2
    blk_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_k)))
    # blocked_k dim3
    contiguous_kv_elems_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_k)))

    q_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=query_grp_sz_layout)
    head_sz_offs = gl.arange(0, HEAD_SZ_POW2, layout=head_sz_layout)
    head_sz_div_offs = gl.arange(0, K_HEAD_SZ_POW2_SPLIT, layout=head_sz_div_layout)
    blk_offs = gl.arange(0, KV_BLK_SZ_POW2, layout=blk_layout)
    contiguous_kv_elems_offs = gl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD, layout=contiguous_kv_elems_layout)

    kv_blk_start = seq_part_idx * MAX_NUM_KV_BLKS
    qk_row_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_mfma_layout))
    qk_col_offs = kv_blk_start + gl.arange(0, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, layout=gl.SliceLayout(0, qk_mfma_layout))

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = gl.zeros([QUERY_GRP_SZ_POW2], dtype=gl.float32)
    else:
        # alibi_slope = gl.load(
        #     alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
        #     mask=q_grp_offs < QUERY_GRP_SZ,
        #     other=0.0,
        # )
        alibi_slope = gl.amd.cdna3.buffer_load(ptr=alibi_slopes + kv_head_idx * QUERY_GRP_SZ, offsets=qk_row_offs, mask=qk_row_offs < QUERY_GRP_SZ)

    # load all kv blocks in one time
    blk_ids = gl.arange(0, MAX_NUM_KV_BLKS, layout=blk_id_layout)
    masked_blk_ids = gl.where(blk_ids < num_kv_blks, blk_ids, 0)
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    # kv_blk_nums = gl.load(blk_tables_start_ptr + kv_blk_start + masked_blk_ids)
    kv_blk_nums = gl.amd.cdna3.buffer_load(ptr=blk_tables_start_ptr + kv_blk_start, offsets=masked_blk_ids)

    # # load q[1, QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    # q_offs = (
    #     seq_idx * stride_q_s
    #     + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[None, :, None]) * stride_q_nh
    #     + head_sz_offs[None, None, :]
    # )
    # q_mask = (q_grp_offs[None, :, None] < QUERY_GRP_SZ) & (head_sz_offs[None, None, :] < HEAD_SZ)
    # # q = gl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    # q = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=q_offs, mask=q_mask)
    # q = (q * scale).to(compute_type)

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    # q = gl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=q_offs, mask=q_mask)
    # q = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=q_offs)
    q = (q * scale).to(compute_type)

    # k_blk_offs[MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, KV_BLK_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD]
    k_blk_offs = (
        kv_blk_nums[:, None, None, None] * stride_k_b
        + kv_head_idx * stride_k_nh
        + head_sz_div_offs[None, :, None, None] * stride_k_hz
        + blk_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
        + contiguous_kv_elems_offs[None, None, None, :]
    )
    # blk_seq_offs = ((kv_blk_start + blk_ids[:, None]) * KV_BLK_SZ + blk_offs[None, :])
    # blk_seq_offs[MAX_NUM_KV_BLKS, 1, KV_BLK_SZ_POW2, 1]
    blk_seq_offs = ((kv_blk_start + blk_ids[:, None, None, None]) * KV_BLK_SZ + blk_offs[None, None, :, None])

    k_mask = (
        # (blk_seq_offs[:, None, :, None] < seq_len) &
        (blk_seq_offs < seq_len) &
        (blk_offs[None, None, :, None] < KV_BLK_SZ) &
        (head_sz_div_offs[None, :, None, None] < (HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD))
    )

    # k[MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, KV_BLK_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD]
    # k_0 = gl.load(k_cache_ptr + k_blk_offs)
    k_0 = gl.amd.cdna3.buffer_load(ptr=k_cache_ptr, offsets=k_blk_offs)
    k = k_0.to(gl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
    k = k.to(compute_type)
    # (MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, KV_BLK_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD) --> (K_HEAD_SZ_POW2_SPLIT, CONTIGUOUS_KV_ELEMS_16B_LOAD, MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2)
    kt_temp = tl.permute(k, [1, 3, 0, 2])
    # k[HEAD_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    kt = tl.reshape(kt_temp, [HEAD_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])
    # # (MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, KV_BLK_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD) --> (MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, CONTIGUOUS_KV_ELEMS_16B_LOAD, KV_BLK_SZ_POW2)
    # kt_temp = tl.permute(k, [0, 1, 3, 2])
    # # k[MAX_NUM_KV_BLKS, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    # kt = tl.reshape(kt_temp, [MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2])

    # # qk[MAX_NUM_KV_BLKS, QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
    # qk = gl.dot(q, kt, out_dtype=gl.float32)
    # qk[QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    # qk = gl.dot(q, k, out_dtype=gl.float32)
    accumulator = gl.zeros((QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2), dtype=gl.float32, layout=qk_mfma_layout)
    qc = gl.convert_layout(q, layout=qk_lhs_layout)
    kc = gl.convert_layout(kt, layout=qk_rhs_layout)
    qk = gl.amd.cdna3.mfma(qc, kc, accumulator)
    # qk = qk.to(compute_type)

    # blk_seq_flatten_offs = gl.reshape(blk_seq_offs, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])
    # if alibi_slopes is not None:
    #     qk += (alibi_slope[:, None] * (blk_seq_flatten_offs - seq_len + 1)[None, :]).to(gl.float32)
    # qk = gl.where(
    #     (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_flatten_offs[None, :] < seq_len),
    #     qk,
    #     float("-inf"),
    # )

    if alibi_slopes is not None:
        qk += (alibi_slope[:, None] * (qk_col_offs - seq_len + 1)[None, :]).to(gl.float32)
    qk = gl.where(
        (qk_row_offs[:, None] < QUERY_GRP_SZ) & (qk_col_offs[None, :] < seq_len),
        qk,
        float("-inf"),
    )

    max_logit_new = gl.max(qk, axis=1)
    # p[QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    # p = gl.math.exp2((qk - max_logit_new[:, None]) * log2e)
    p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
    exp_sum = gl.sum(p, axis=1)
    p = p.to(compute_type)

    m_l_base_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_mfma_layout))
    m_l_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + m_l_base_offs
    )
    m_l_grp_mask = m_l_base_offs < QUERY_GRP_SZ
    gl.amd.cdna3.buffer_store(stored_value=max_logit_new, ptr=max_logits_ptr, offsets=m_l_offs, mask=m_l_grp_mask)
    gl.amd.cdna3.buffer_store(stored_value=exp_sum, ptr=exp_sums_ptr, offsets=m_l_offs, mask=m_l_grp_mask)


    # # shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
    # # stride_logits_s,
    # # stride_logits_nh,
    # # stride_logits_p,
    # # stride_logits_g,
    # o_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_mfma_layout))
    # o_head_sz_offs = gl.arange(0, 256, layout=gl.SliceLayout(0, qk_mfma_layout))
    # o_mask = (o_grp_offs[:, None] < QUERY_GRP_SZ) & (kv_blk_start + o_head_sz_offs[None, :] < seq_len)
    # logits_offs = seq_idx * stride_logits_s
    # logits_offs += kv_head_idx * stride_logits_nh
    # logits_offs += (
    #     seq_part_idx * stride_logits_p
    #     + o_grp_offs[:, None] * stride_logits_g
    #     + o_head_sz_offs[None, :]
    # )
    # gl.amd.cdna3.buffer_store(stored_value=p, ptr=logits_ptr, offsets=logits_offs, mask=o_mask)
    # # gl.amd.cdna3.buffer_store(stored_value=qk.to(compute_type), ptr=logits_ptr, offsets=logits_offs, mask=o_mask)
    # # gl.amd.cdna3.buffer_store(stored_value=qk.to(compute_type), ptr=logits_ptr, offsets=logits_offs)




    # # v_blk_offs[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    # v_blk_offs = (
    #     kv_blk_nums[:, None, None] * stride_v_b
    #     + kv_head_idx * stride_v_nh
    #     + head_sz_offs[None, :, None] * stride_v_hz
    #     + blk_offs[None, None, :]
    # )
    # v_mask = (
    #     (blk_seq_offs[:, None, :] < seq_len) &
    #     (blk_offs[None, None, :] < KV_BLK_SZ) &
    #     (head_sz_offs[None, :, None] < HEAD_SZ)
    # )

    # MAX_NUM_KV_BLKS x HEAD_SZ_POW2 x KV_BLK_SZ_POW2
    # 16(kdim0) x 128(ndim) x 16(kdim1)
    blocked_v_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[4, 4,  8],
        threads_per_warp=[1, 32, 2],
        warps_per_cta   =[4, 1,  1],
        order           =[2, 1,  0],
    )
    v_dim0_offs = gl.arange(0, MAX_NUM_KV_BLKS, layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_v_layout)))
    v_dim1_offs = gl.arange(0, HEAD_SZ_POW2, layout=gl.SliceLayout(0, gl.SliceLayout(2, blocked_v_layout)))
    v_dim2_offs = gl.arange(0, KV_BLK_SZ_POW2, layout=gl.SliceLayout(0, gl.SliceLayout(1, blocked_v_layout)))

    # # load all kv blocks in one time
    # blk_ids = gl.arange(0, MAX_NUM_KV_BLKS, layout=blk_id_layout)
    # masked_blk_ids = gl.where(blk_ids < num_kv_blks, blk_ids, 0)
    # blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    # # kv_blk_nums = gl.load(blk_tables_start_ptr + kv_blk_start + masked_blk_ids)
    # kv_blk_nums = gl.amd.cdna3.buffer_load(ptr=blk_tables_start_ptr + kv_blk_start, offsets=masked_blk_ids)

    kv_blk_nums2 = gl.convert_layout(kv_blk_nums, layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_v_layout)))
    # v_blk_offs[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    v_blk_offs = (
        kv_blk_nums2[:, None, None] * stride_v_b
        + kv_head_idx * stride_v_nh
        + v_dim1_offs[None, :, None] * stride_v_hz
        + v_dim2_offs[None, None, :]
    )
    v_len_offs = kv_blk_start + v_dim0_offs[:, None, None] * KV_BLK_SZ + v_dim2_offs[None, None, :]
    v_mask = (
        (v_len_offs < seq_len) &
        (v_dim2_offs[None, None, :] < KV_BLK_SZ) &
        (v_dim1_offs[None, :, None] < HEAD_SZ)
    )

    # v[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    # v_0 = gl.load(v_cache_ptr + v_blk_offs)
    v_0 = gl.amd.cdna3.buffer_load(ptr=v_cache_ptr, offsets=v_blk_offs)
    # v = v_0.to(gl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
    v = v_0.to(compute_type)
    # [MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2] --> [MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2, HEAD_SZ_POW2]
    v = gl.permute(v, [0, 2, 1])
    # v[MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2]
    v = gl.reshape(v, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2])

    m_l_base_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_mfma_layout))
    m_l_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + m_l_base_offs
    )
    m_l_grp_mask = m_l_base_offs < QUERY_GRP_SZ

    # q_grp_offs2 = tl.arange(0, QUERY_GRP_SZ_POW2)
    # max_logits_offs = (
    #     seq_idx * stride_max_logits_s
    #     + kv_head_idx * stride_max_logits_nh
    #     + seq_part_idx * stride_max_logits_p
    #     + q_grp_offs2
    # )
    # m_grp_mask = q_grp_offs2 < QUERY_GRP_SZ
    # tl.store(max_logits_ptr + max_logits_offs, max_logit_new, mask=m_grp_mask)
    # tl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)
    gl.amd.cdna3.buffer_store(stored_value=max_logit_new, ptr=max_logits_ptr, offsets=m_l_offs, mask=m_l_grp_mask)
    gl.amd.cdna3.buffer_store(stored_value=exp_sum, ptr=exp_sums_ptr, offsets=m_l_offs, mask=m_l_grp_mask)

    # acc[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    # acc = gl.dot(p, v, out_dtype=gl.float32)
    # acc = acc / exp_sum[:, None]
    accumulator2 = gl.zeros((QUERY_GRP_SZ_POW2, HEAD_SZ_POW2), dtype=gl.float32, layout=pv_mfma_layout)
    pc = gl.convert_layout(p, layout=pv_lhs_layout)
    vc = gl.convert_layout(v, layout=pv_rhs_layout)
    acc = gl.amd.cdna3.mfma(pc, vc, accumulator2)
    exp_sum = gl.convert_layout(exp_sum[:, None], layout=pv_mfma_layout)
    exp_sum = tl.broadcast_to(exp_sum, QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
    # exp_sum = tl.broadcast_to(exp_sum[:, None], QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
    # exp_sum = gl.convert_layout(exp_sum, layout=pv_mfma_layout)
    acc = acc / exp_sum
    acc = acc.to(compute_type)

    # end up computation
    o_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, pv_mfma_layout))
    o_head_sz_offs = gl.arange(0, HEAD_SZ_POW2, layout=gl.SliceLayout(0, pv_mfma_layout))
    o_mask = (o_grp_offs[:, None] < QUERY_GRP_SZ) & (o_head_sz_offs[None, :] < HEAD_SZ)
    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + o_grp_offs[:, None] * stride_logits_g
        + o_head_sz_offs[None, :]
    )
    # gl.store(logits_ptr + logits_offs, acc, mask=q_mask)
    gl.amd.cdna3.buffer_store(stored_value=acc, ptr=logits_ptr, offsets=logits_offs, mask=o_mask)


# Compile ttgir with triton
def compile_ttgir_with_triton(ttgir_content: str):
    # Read ttgir string from file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir', delete=False) as f:
        f.write(ttgir_content)
        ttgir_path = f.name

    # Compile ttgir string to ttgir
    try:
        compiled_artifact = compile(ttgir_path)
        return compiled_artifact
    finally:
        os.unlink(ttgir_path)

@perftest()
# @perftest(num_iters=TEST_NUM_ITERS)
def _paged_attn_decode_v2_w_dot_kernel_reshape_wrapper(
    grid,
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
    stride_v_hz,
    stride_v_bz,
    stride_bt_s,
    kv_type,
    compute_type,
    HEAD_SZ,
    HEAD_SZ_POW2,
    QUERY_GRP_SZ,
    QUERY_GRP_SZ_POW2,
    KV_BLK_SZ,
    KV_BLK_SZ_POW2,
    SEQ_PARTITION_SZ,
):
    # Use ttgir as input
    # print(f"_paged_attn_decode_v2_w_dot_kernel_reshape_wrapper")
    # import pdb
    # pdb.set_trace()

    # if 1:
    if 0:
        with open(ttgir_file_path, 'r') as f:
            ttgir_content = f.read()

        try:
            compiled_kernel = compile_ttgir_with_triton(ttgir_content)
            compiled_kernel[grid](
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
                stride_v_hz,
                stride_v_bz,
                stride_bt_s,
            )
        except Exception as e:
            print(f"Compilation failed: {e}")
    else:
        # _paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk[grid](
        _paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk_gluon[grid](
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
            stride_v_hz,
            stride_v_bz,
            stride_bt_s,
            compute_type=compute_type,
            HEAD_SZ=HEAD_SZ,
            HEAD_SZ_POW2=HEAD_SZ_POW2,
            QUERY_GRP_SZ=QUERY_GRP_SZ,
            QUERY_GRP_SZ_POW2=QUERY_GRP_SZ_POW2,
            KV_BLK_SZ=KV_BLK_SZ,
            KV_BLK_SZ_POW2=KV_BLK_SZ_POW2,
            SEQ_PARTITION_SZ=SEQ_PARTITION_SZ,
        )

@perftest()
def _paged_attn_decode_v2_w_dot_reduce_kernel_wrapper(
    grid,
    out_ptr,        # [num_seqs, num_kv_heads, q_grp_sz, head_sz]
    exp_sums_ptr,   # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr, # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptrs,    # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    seq_lens_ptr,   # [num_seqs]
    stride_o_s,
    stride_o_h,
    stride_exp_sums_s,
    stride_exp_sums_h,
    stride_exp_sums_p,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_logits_g,
    HEAD_SZ,
    HEAD_SZ_POW2,
    QUERY_GRP_SZ,
    QUERY_GRP_SZ_POW2,
    SEQ_PARTITION_SZ,
    MAX_NUM_SEQ_PARTITIONS,
    MAX_NUM_SEQ_PARTITIONS_POW2,
):
    _paged_attn_decode_v2_w_dot_reduce_kernel[grid](
        out_ptr,        # [num_seqs, num_kv_heads, q_grp_sz, head_sz]
        exp_sums_ptr,   # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
        max_logits_ptr, # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
        logits_ptrs,    # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
        seq_lens_ptr,   # [num_seqs]
        stride_o_s,
        stride_o_h,
        stride_exp_sums_s,
        stride_exp_sums_h,
        stride_exp_sums_p,
        stride_logits_s,
        stride_logits_h,
        stride_logits_p,
        stride_logits_g,
        HEAD_SZ=HEAD_SZ,
        HEAD_SZ_POW2=HEAD_SZ_POW2,
        QUERY_GRP_SZ=QUERY_GRP_SZ,
        QUERY_GRP_SZ_POW2=QUERY_GRP_SZ_POW2,
        SEQ_PARTITION_SZ=SEQ_PARTITION_SZ,
        MAX_NUM_SEQ_PARTITIONS=MAX_NUM_SEQ_PARTITIONS,
        MAX_NUM_SEQ_PARTITIONS_POW2=MAX_NUM_SEQ_PARTITIONS_POW2,
    )

def paged_attention_decode(
    output: torch.Tensor,       # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,        # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,    # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    seq_lens: torch.Tensor,     # [num_seqs]
    block_tables: torch.Tensor, # [num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    compute_type,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_seq_partitions: int = 0,  # TODO use this below
    alibi_slopes: torch.Tensor = None,
) -> None:
    """
    #TODO: Add Doc
    """

    # get num_seqs, num_kv_heads, kv_blk_sz, head_sz and query_grp_sz
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]

    max_num_partitions = int((max_seq_len + _SEQ_PARTITION_SIZE - 1) // _SEQ_PARTITION_SIZE)

    # use_v1 = max_seq_len <= 8192 and (
    #     max_num_partitions == 1 or num_seqs * num_q_heads > 512
    # )
    use_v1 = False
    print("***************************************************")
    print(f"k_scale.numel()={k_scale.numel()}")
    print("***************************************************")
    if k_scale.numel() > 1:
        if use_v1:
            paged_attn_decode_v1_per_token_quant(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale,
                v_scale,
            )
        else:
            paged_attn_decode_v2_per_token_quant(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale,
                v_scale,
                max_num_partitions,
            )
    else:
        if use_v1:
            paged_attn_decode_v1(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale.item(),
                v_scale.item(),
            )
        else:
            perf_time = paged_attn_decode_v2(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale.item(),
                v_scale.item(),
                max_num_partitions,
            )
            return perf_time


def paged_attn_decode_v1(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: float,
    v_scale: float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = query.shape[1] // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA- Multi-Head Attention
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v1_wo_dot_kernel[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            query.stride(0),
            query.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            MAX_SEQ_LEN_POW2=max_seq_len,
        )
    # GQA - Grouped Query Attention
    else:
        grid = (num_seqs, num_kv_heads, 1)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v1_w_dot_kernel[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz,
        )


@triton.jit
def _paged_attn_decode_v1_wo_dot_kernel(
    out,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    q_ptr,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    blk_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_q_heads]
    scale,
    k_scale,
    v_scale,
    stride_q_s,
    stride_q_h,
    stride_o_s,
    stride_o_nh,
    stride_o_hs,
    stride_k_b,
    stride_k_nh,
    stride_k_kb,
    stride_bt_s,
    compute_type: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    MAX_SEQ_LEN_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    head_idx = tl.program_id(axis=0)
    seq_idx = tl.program_id(axis=1)
    kv_head_idx = head_idx // QUERY_GRP_SZ

    log2e: tl.constexpr = 1.4426950408889634

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    num_kv_blks = tl.cdiv(seq_len, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)

    # load alibi slopes [1]
    if alibi_slopes_ptr is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes_ptr + head_idx)

    # load q [1, HEAD_SZ_POW2]
    q_offs = seq_idx * stride_q_s + head_idx * stride_q_h + head_sz_offs
    q = tl.load(q_ptr + q_offs, mask=head_sz_offs < HEAD_SZ)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([KV_BLK_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = float("-inf")
    exp_sum = 0.0

    kv_offs = (
        kv_head_idx * stride_k_nh
        + blk_offs[:, None] * stride_k_kb
        + head_sz_offs[None, :]
    )
    blk_tbl_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_nums = tl.load(blk_tbl_start_ptr + b)
        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = b * KV_BLK_SZ + blk_offs
        kv_mask = (
            (blk_seq_offs[:, None] < seq_len)
            & (blk_offs[:, None] < KV_BLK_SZ)
            & (head_sz_offs[None, :] < HEAD_SZ)
        )

        # load k [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        if k_0.dtype.is_fp8():
            k = k_0.to(tl.float32) * k_scale
        else:
            k = k_0
        k = k.to(compute_type)

        # qk #[KV_BLK_SZ_POW2]
        qk = tl.sum(
            (q[None, :] * k).to(tl.float32), axis=1
        )  # [1, HEAD_SZ_POW2] * [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        if alibi_slopes_ptr is not None:
            qk += (alibi_slope * (blk_seq_offs - seq_len + 1)).to(tl.float32)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        max_logit_new = tl.maximum(tl.max(qk, axis=0), max_logit)

        # p: [KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # load v [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask)
        if v_0.dtype.is_fp8():
            v = v_0.to(tl.float32) * v_scale
        else:
            v = v_0
        v = v.to(compute_type)

        acc += p[:, None] * v

        exp_sum = exp_sum * alpha + tl.sum(p, axis=0)
        max_logit = max_logit_new

    acc = acc / exp_sum

    offs_out = seq_idx * stride_o_s + head_idx * stride_o_nh + head_sz_offs
    out_mask = head_sz_offs < HEAD_SZ
    tl.store(
        out + offs_out, tl.sum(acc, axis=0).to(out.dtype.element_ty), mask=out_mask
    )


@triton.jit
def _paged_attn_decode_v1_w_dot_kernel(
    out_ptr,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    q_ptr,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,  # [num_blocks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr,  # [num_blocks, num_kv_heads, kv_blk_sz, head_sz]
    blk_tables_ptr,  # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes,  # [num_kv_heads*query_grp_sz]
    scale,
    k_scale,
    v_scale,
    stride_o_s,
    stride_o_nh,
    stride_o_hs,
    stride_q_s,
    stride_q_nh,
    stride_q_hs,
    stride_k_b,
    stride_k_nh,
    stride_k_kb,
    stride_k_hs,
    stride_bt_s,
    stride_bt_nb,
    compute_type: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    log2e: tl.constexpr = 1.4426950408889634

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    num_kv_blks = tl.cdiv(seq_len, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :] * stride_q_hs
    )

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)

    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)

    kv_offs = (
        kv_head_idx * stride_k_nh
        + blk_offs[:, None] * stride_k_kb
        + head_sz_offs[None, :] * stride_k_hs
    )
    blk_tbl_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_nums = tl.load(blk_tbl_start_ptr + b)
        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = b * KV_BLK_SZ + blk_offs
        kv_mask = (
            (blk_seq_offs[:, None] < seq_len)
            & (blk_offs[:, None] < KV_BLK_SZ)
            & (head_sz_offs[None, :] < HEAD_SZ)
        )

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
        k = k.to(compute_type)

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (blk_seq_offs - seq_len + 1)[None, :]).to(
                tl.float32
            )

        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )
        max_logit_new = tl.maximum(tl.max(qk, axis=1), max_logit)

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ, HEAD_SZ]
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
        v = v.to(compute_type)

        p = p.to(v.dtype)
        acc += tl.dot(p, v, out_dtype=tl.float32)

        exp_sum = exp_sum * alpha + tl.sum(p, axis=1)
        max_logit = max_logit_new

    acc = acc / exp_sum[:, None]

    out_offs = (
        seq_idx * stride_o_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_o_nh
        + head_sz_offs[None, :]
    )

    out_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def paged_attn_decode_v2(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq],
    seq_lens: torch.Tensor,  # [num_seqs],
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: float,
    v_scale: float,
    max_num_partitions: int,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = value_cache.shape[3]
    head_sz = value_cache.shape[2]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)

    # Note: There is a bug in triton.next_power_of_2 function which causes it
    # to update the passed in arg, so that's why we have a workaround here
    # max_num_partitions_pow2 = triton.next_power_of_2(max_num_partitions)
    if max_num_partitions == 0:
        max_num_partitions_pow2 = 1
    else:
        max_num_partitions_pow2 = 2 ** math.ceil(math.log2(max_num_partitions))

    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, max_num_partitions)
        shape_info = (num_seqs, num_q_heads, max_num_partitions)
        exp_sums = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        max_logits = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        tmp_output = torch.empty(
            (*shape_info, head_sz), dtype=output.dtype, device=output.device
        )
        _paged_attn_decode_v2_wo_dot_kernel[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            block_tables.stride(1),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_BLKS_PER_SEQ=block_tables.shape[1],
            MAX_SEQ_LEN_POW2=max_seq_len,
        )
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v2_wo_dot_reduce_kernel[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS=int(max_num_partitions),
            MAX_NUM_SEQ_PARTITIONS_POW2=int(max_num_partitions_pow2),
        )
    # GQA
    else:
        grid = (num_seqs, num_kv_heads, max_num_partitions)
        shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
        max_logits = torch.zeros(shape_info, dtype=torch.float32, device=output.device)
        exp_sums = torch.zeros(shape_info, dtype=torch.float32, device=output.device)
        # tmp_output = torch.empty(
        tmp_output = torch.zeros(
            # (*shape_info, 256), dtype=output.dtype, device=output.device
            *shape_info, head_sz, dtype=output.dtype, device=output.device
        )
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)

        compute_type2 = gl.bfloat16
        print(f"grid={grid}")
        print(f"query.shape={query.shape}")
        print(f"key_cache.shape={key_cache.shape}")
        print(f"value_cache.shape={value_cache.shape}")
        print(f"output.shape={output.shape}")
        print(f"block_tables.shape={block_tables.shape}")
        print(f"query.dtype={query.dtype}")
        print(f"key_cache.dtype={key_cache.dtype}")
        print(f"value_cache.dtype={value_cache.dtype}")
        print(f"output.dtype={output.dtype}")
        print(f"block_tables.dtype={block_tables.dtype}")
        input_config = dict(
            scale=scale,
            max_num_partitions=max_num_partitions,
            kv_type=compute_type,
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            stride_logits_s=tmp_output.stride(0),
            stride_logits_nh=tmp_output.stride(1),
            stride_logits_p=tmp_output.stride(2),
            stride_logits_g=tmp_output.stride(3),
        )
        print(input_config)

        _, decode_time = _paged_attn_decode_v2_w_dot_kernel_reshape_wrapper(
            grid,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            block_tables.stride(0),
            kv_type=compute_type2,
            compute_type=compute_type2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        )
        grid = (num_seqs, num_kv_heads, 1)
        _, reduce_time = _paged_attn_decode_v2_w_dot_reduce_kernel_wrapper(
            grid,
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS=int(max_num_partitions),
            MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions)),
        )
        # reduce_time = 0
        # print(f"gluon:\n{tmp_output[0]}")

        return {'triton_decode': decode_time,
                'triton_reduce': reduce_time,
                'tmp_output': tmp_output,
                'exp_sums': exp_sums,
                'max_logits': max_logits,
                'triton': decode_time + reduce_time}


@triton.jit
def _paged_attn_decode_v2_wo_dot_kernel(
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    blk_tables_ptr,
    seq_lens_ptr,
    scale,
    k_scale,
    v_scale,
    alibi_slopes,
    stride_exp_s,
    stride_exp_h,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_q_s,
    stride_q_h,
    stride_k_b,
    stride_k_nh,
    stride_k_kb,
    stride_bt_s,
    stride_bt_nb,
    compute_type: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_BLKS_PER_SEQ: tl.constexpr,
    MAX_SEQ_LEN_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    seq_part_idx = tl.program_id(2)
    kv_head_idx = head_idx // QUERY_GRP_SZ

    log2e: tl.constexpr = 1.4426950408889634

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if seq_part_idx * SEQ_PARTITION_SZ >= seq_len:
        return

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)

    # load alibi slopes
    if alibi_slopes is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes + head_idx)

    # load q[HEAD_SZ]
    q_offs = seq_idx * stride_q_s + head_idx * stride_q_h + head_sz_offs
    q = tl.load(q_ptr + q_offs, mask=head_sz_offs < HEAD_SZ)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([KV_BLK_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = float("-inf")
    exp_sum = 0.0

    kv_offs = (
        kv_head_idx * stride_k_nh
        + blk_offs[:, None] * stride_k_kb
        + head_sz_offs[None, :]
    )
    kv_blk_start = seq_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    blk_tables_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_idx = kv_blk_start + b
        kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_idx * stride_bt_nb)

        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = kv_blk_idx * KV_BLK_SZ + blk_offs
        kv_mask = (
            (blk_seq_offs[:, None] < seq_len)
            & (blk_offs[:, None] < KV_BLK_SZ)
            & (head_sz_offs[None, :] < HEAD_SZ)
        )

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
        k = k.to(compute_type)

        # qk: [KV_BLK_SZ_POW2]
        qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        if alibi_slopes is not None:
            qk += (alibi_slope * (blk_seq_offs - seq_len + 1)).to(tl.float32)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        max_logit_new = tl.maximum(max_logit, tl.max(qk, axis=0))

        # p: [KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
        v = v.to(compute_type)

        # acc: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        acc += p[:, None] * v

        exp_sum = exp_sum * alpha + tl.sum(p, axis=0)
        max_logit = max_logit_new

    acc = acc / exp_sum

    max_logits_offs = seq_idx * stride_exp_s + head_idx * stride_exp_h + seq_part_idx

    tl.store(max_logits_ptr + max_logits_offs, max_logit)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum)

    logits_offs = (
        seq_idx * stride_logits_s
        + head_idx * stride_logits_h
        + seq_part_idx * stride_logits_p
        + head_sz_offs
    )
    logits_mask = head_sz_offs < HEAD_SZ
    tl.store(
        logits_ptr + logits_offs,
        tl.sum(acc, axis=0).to(logits_ptr.dtype.element_ty),
        mask=logits_mask,
    )


@triton.jit
def _paged_attn_decode_v2_wo_dot_reduce_kernel(
    out,
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptr,
    seq_lens,
    stride_out_n,
    stride_out_h,
    stride_exp_sums_n,
    stride_exp_sums_h,
    stride_logits_n,
    stride_logits_h,
    stride_logits_b,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    # get seq_idx, head_idx, seq_len
    head_idx = tl.program_id(axis=0)
    seq_idx = tl.program_id(axis=1)

    seq_len = tl.load(seq_lens + seq_idx)
    num_partitions = tl.cdiv(seq_len, SEQ_PARTITION_SZ)

    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    seq_part_offs = tl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2)

    max_logit = float("-inf")
    acc = tl.zeros([HEAD_SZ_POW2], dtype=tl.float32)
    global_exp_sum = tl.zeros([1], dtype=tl.float32)

    # load max_logits [MAX_NUM_SEQ_PARTITIONS_POW2]
    max_logits_offs = (
        seq_idx * stride_exp_sums_n + head_idx * stride_exp_sums_h + seq_part_offs
    )
    max_logits_mask = seq_part_offs < num_partitions
    max_logits = tl.load(
        max_logits_ptr + max_logits_offs,
        mask=max_logits_mask,
        other=float("-inf"),
    )

    # find max_logit
    max_logit = tl.max(max_logits, axis=0)

    # load exp_sum [MAX_NUM_SEQ_PARTITIONS_POW2]
    exp_sums_offs = (
        seq_idx * stride_exp_sums_n + head_idx * stride_exp_sums_h + seq_part_offs
    )
    exp_sums_mask = seq_part_offs < num_partitions
    exp_sums = tl.load(
        exp_sums_ptr + exp_sums_offs,
        mask=exp_sums_mask,
        other=0.0,
    )

    # rescaled_exp_sum and global_exp_sum
    # [MAX_NUM_SEQ_PARTITIONS_POW2]
    rescaled_exp_sum = exp_sums * tl.exp(max_logits - max_logit)
    global_exp_sum += tl.sum(rescaled_exp_sum, axis=0)
    rescaled_exp_sum /= global_exp_sum

    # load logits
    logits_offs = (
        seq_idx * stride_logits_n
        + head_idx * stride_logits_h
        + seq_part_offs[:, None] * stride_logits_b
        + head_sz_offs[None, :]
    )
    logits_mask = (seq_part_offs[:, None] < num_partitions) & (
        head_sz_offs[None, :] < HEAD_SZ
    )

    logits = tl.load(logits_ptr + logits_offs, mask=logits_mask, other=0.0)
    acc += tl.sum(logits * rescaled_exp_sum[:, None], axis=0)

    # store the final output
    out_ptr = seq_idx * stride_out_n + head_idx * stride_out_h + head_sz_offs
    out_mask = head_sz_offs < HEAD_SZ
    tl.store(out + out_ptr, acc.to(out.dtype.element_ty), mask=out_mask)


@triton.jit
def _paged_attn_decode_v2_w_dot_kernel(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    q_ptr,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    blk_tables_ptrs,  # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,  # [num_seqs]
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
    stride_k_kb,
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

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if seq_part_idx * SEQ_PARTITION_SZ >= seq_len:
        return

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)

    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)

    kv_offs = (
        kv_head_idx * stride_k_nh
        + blk_offs[:, None] * stride_k_kb
        + head_sz_offs[None, :]
    )
    kv_blk_start = seq_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    for b in range(num_kv_blks):
        kv_blk_idx = kv_blk_start + b
        kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_idx)

        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = kv_blk_idx * KV_BLK_SZ + blk_offs
        kv_mask = (
            (blk_seq_offs[:, None] < seq_len)
            & (blk_offs[:, None] < KV_BLK_SZ)
            & (head_sz_offs[None, :] < HEAD_SZ)
        )

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
        k = k.to(compute_type)

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (blk_seq_offs - seq_len + 1)[None, :]).to(
                tl.float32
            )
        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        max_logit_new = tl.maximum(max_logit, tl.max(qk, axis=1))

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
        v = v.to(compute_type)

        p = p.to(v.dtype)
        acc += tl.dot(p, v, out_dtype=tl.float32)

        exp_sum = exp_sum * alpha + tl.sum(p, axis=1)
        max_logit = max_logit_new

    acc = acc / exp_sum[:, None]

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs, max_logit, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)

    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )

    tl.store(logits_ptr + logits_offs, acc, mask=q_mask)

# Transpose k_cache_ptr before execution
# Suffer from k_cache loading without prefetching (131 us)
@triton.jit
def _paged_attn_decode_v2_w_dot_kernel_reshape(
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
    stride_v_kb,
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

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if seq_part_idx * SEQ_PARTITION_SZ >= seq_len:
        return

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)

    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)
    max_num_kv_blks: tl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    head_sz_div_offs = tl.arange(0, HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
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

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)

    # k_offs[HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_offs = (
        kv_head_idx * stride_k_nh
        + head_sz_div_offs[:, None, None] * stride_k_hz
        + blk_offs[None, :, None] * stride_k_bz
        + contiguous_kv_elems_offs[None, None, :]
    )
    # v_offs[HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    v_offs = (
        kv_head_idx * stride_v_nh
        + blk_offs[None, :]                     # blk_offs: [KV_BLK_SZ_POW2]
        + head_sz_offs[:, None] * stride_v_kb   # head_sz_offs: [HEAD_SZ_POW2]
    )
    kv_blk_start = seq_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    for b in range(num_kv_blks):
        kv_blk_idx = kv_blk_start + b
        kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_idx)

        k_blk_offs = kv_blk_nums * stride_k_b + k_offs
        blk_seq_offs = kv_blk_idx * KV_BLK_SZ + blk_offs
        k_mask = (
            (blk_seq_offs[None, :, None] < seq_len)   # blk_seq_offs: [KV_BLK_SZ_POW2]
            & (blk_offs[None, :, None] < KV_BLK_SZ)   # blk_offs: [KV_BLK_SZ_POW2]
            & (head_sz_div_offs[:, None, None] < (HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD)) # head_sz_offs: [HEAD_SZ_POW2]
        )

        # load k[HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
        k_0 = tl.load(k_cache_ptr + k_blk_offs, mask=k_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
        k = k.to(compute_type)
        # k[HEAD_SZ_POW2, KV_BLK_SZ_POW2]
        k = tl.permute(k, [0, 2, 1]) # [HEAD_SZ_POW2/x, x, KV_BLK_SZ_POW2]
        k = tl.reshape(k, [HEAD_SZ_POW2, KV_BLK_SZ_POW2])

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        qk = tl.dot(q, k, out_dtype=tl.float32)
        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (blk_seq_offs - seq_len + 1)[None, :]).to(
                tl.float32
            )
        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        max_logit_new = tl.maximum(max_logit, tl.max(qk, axis=1))

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # load v[HEAD_SZ_POW2, KV_BLK_SZ_POW2]
        v_blk_offs = kv_blk_nums * stride_v_b + v_offs
        v_mask = (
            (blk_seq_offs[None, :] < seq_len)   # blk_seq_offs: [KV_BLK_SZ_POW2]
            & (blk_offs[None, :] < KV_BLK_SZ)   # blk_offs: [KV_BLK_SZ_POW2]
            & (head_sz_offs[:, None] < HEAD_SZ) # head_sz_offs: [HEAD_SZ_POW2]
        )
        v_0 = tl.load(v_cache_ptr + v_blk_offs, mask=v_mask, other=0.0)
        v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
        v = v.to(compute_type)
        # v[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v = tl.permute(v, [1, 0])

        p = p.to(v.dtype)
        acc += tl.dot(p, v, out_dtype=tl.float32)

        exp_sum = exp_sum * alpha + tl.sum(p, axis=1)
        max_logit = max_logit_new

    acc = acc / exp_sum[:, None]

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs, max_logit, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)

    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )

    tl.store(logits_ptr + logits_offs, acc, mask=q_mask)

# Reshape KV cache before execution
# Suffer from k_cache loading without prefetching (177 us)
@triton.jit
def _paged_attn_decode_v2_w_dot_kernel_reshape_noloop(
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

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    max_num_kv_blks: tl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)
    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    head_sz_div_offs = tl.arange(0, HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
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
    blk_ids = tl.arange(0, max_num_kv_blks)
    masked_blk_ids = tl.where(blk_ids < num_kv_blks, blk_ids, 0)
    kv_blk_start = seq_part_idx * max_num_kv_blks
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

    # k_blk_offs[max_num_kv_blks, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_blk_offs = (
        kv_blk_nums[:, None, None, None] * stride_k_b
        + kv_head_idx * stride_k_nh
        + head_sz_div_offs[None, :, None, None] * stride_k_hz
        + blk_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
        + contiguous_kv_elems_offs[None, None, None, :]
    )
    # blk_seq_offs[max_num_kv_blks, KV_BLK_SZ_POW2]
    blk_seq_offs = ((kv_blk_start + blk_ids[:, None]) * KV_BLK_SZ  # blk_ids: [max_num_kv_blks]
                    + blk_offs[None, :]) # blk_offs: [KV_BLK_SZ_POW2]

    k_mask = (
        (blk_seq_offs[:, None, :, None] < seq_len) &
        (blk_offs[None, None, :, None] < KV_BLK_SZ) &
        (head_sz_div_offs[None, :, None, None] < (HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD))
    )

    # k[max_num_kv_blks, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_0 = tl.load(k_cache_ptr + k_blk_offs)
    k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
    k = k.to(compute_type)
    # k[HEAD_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2]
    k = tl.permute(k, [0, 2, 1, 3]) # [max_num_kv_blks, KV_BLK_SZ_POW2, HEAD_SZ_POW2/x, x]
    k = tl.reshape(k, [max_num_kv_blks * KV_BLK_SZ_POW2, HEAD_SZ_POW2])

    # qk[max_num_kv_blks * KV_BLK_SZ_POW2, QUERY_GRP_SZ_POW2]
    qk = tl.dot(k, q.T, out_dtype=tl.float32)
    blk_seq_flatten_offs = tl.reshape(blk_seq_offs, [max_num_kv_blks * KV_BLK_SZ_POW2])
    if alibi_slopes is not None:
        qk += (alibi_slope[None, :] * (blk_seq_flatten_offs - seq_len + 1)[:, None]).to(
            tl.float32
        )
    qk = tl.where(
        (blk_seq_flatten_offs[:, None] < seq_len),
        qk,
        float("-inf"),
    )

    max_logit_new = tl.max(qk, axis=0)
    p = tl.math.exp2((qk - max_logit_new[None, :]) * log2e)

    # acc[HEAD_SZ_POW2, QUERY_GRP_SZ_POW2]
    p = p.to(compute_type)
    exp_sum = tl.sum(p, axis=0)

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs, max_logit_new, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)

    # v_blk_offs[max_num_kv_blks, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
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
    v_0 = tl.load(v_cache_ptr + v_blk_offs)
    v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
    v = v.to(compute_type)
    # v[HEAD_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2]
    v = tl.permute(v, [1, 0, 2])
    v = tl.reshape(v, [HEAD_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2])
    acc = tl.dot(v, p, out_dtype=tl.float32).T
    acc = acc / exp_sum[:, None]

    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )

    tl.store(logits_ptr + logits_offs, acc, mask=q_mask)

# Reshape KV cache before execution
# Suffer from k_cache loading without prefetching (177 us)
@triton.jit
def _paged_attn_decode_v2_w_dot_kernel_reshape_load4_qk(
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    blk_tables_ptrs,
    seq_lens_ptr,
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
    stride_v_kb,
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
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_part_idx = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634
    CONTIGUOUS_KV_ELEMS_16B_LOAD: tl.constexpr = 8
    LANE16_SIZE: tl.constexpr = 16
    NUM_WARPS: tl.constexpr = 4
    TOKENS_PER_ITER: tl.constexpr = LANE16_SIZE * NUM_WARPS
    KV_BLK_PER_ITER: tl.constexpr = (TOKENS_PER_ITER + KV_BLK_SZ - 1) // KV_BLK_SZ
    NUM_ITER: tl.constexpr = (SEQ_PARTITION_SZ + TOKENS_PER_ITER - 1) // TOKENS_PER_ITER
    MAX_NUM_KV_BLKS: tl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    MAX_SEQ_LEN = (tl.num_programs(2) * SEQ_PARTITION_SZ) // KV_BLK_SZ
    MAX_NUM_KV_BLKS_IN_SEQ = (MAX_SEQ_LEN + KV_BLK_SZ - 1) // KV_BLK_SZ

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if seq_part_idx * SEQ_PARTITION_SZ >= seq_len:
        return

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)
    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    head_sz_div_offs = tl.arange(0, HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    contiguous_kv_elems_offs = tl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD)
    kv_blk_per_iter_offs = tl.arange(0, KV_BLK_PER_ITER)
    tokens_per_iter_offs = tl.arange(0, TOKENS_PER_ITER)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)

    # k_offs[KV_BLK_PER_ITER, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_offs = (
        kv_head_idx * stride_v_nh
        + head_sz_div_offs[:, None, None] * stride_k_hz
        + blk_offs[None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
        + contiguous_kv_elems_offs[None, None, :]
    )
    # v_offs[HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    v_offs = (
        kv_head_idx * stride_v_nh
        + blk_offs[None, :]   # blk_offs: [KV_BLK_SZ_POW2]
        + head_sz_offs[:, None] * stride_v_kb             # head_sz_offs: [HEAD_SZ_POW2]
    )
    kv_blk_start = seq_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    for iter in range(NUM_ITER):
        kv_blk_idx = iter * KV_BLK_PER_ITER + kv_blk_per_iter_offs
        kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_start + kv_blk_idx, mask=(kv_blk_idx < num_kv_blks), other=0)

        k_blk_offs = kv_blk_nums[:, None, None, None] * stride_k_b + k_offs[None, :, :, :]
        blk_seq_offs = kv_blk_idx[:, None] * KV_BLK_SZ + blk_offs[None, :]
        k_mask = (
            (blk_seq_offs[:, None, :, None] < seq_len)   # blk_seq_offs: [KV_BLK_SZ_POW2]
            & (blk_offs[None, None, :, None] < KV_BLK_SZ)   # blk_offs: [KV_BLK_SZ_POW2]
            & (head_sz_div_offs[None, :, None, None] < (HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD)) # head_sz_offs: [HEAD_SZ_POW2]
        )

        # load k[KV_BLK_PER_ITER, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
        k_0 = tl.load(k_cache_ptr + k_blk_offs)
        k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
        k = k.to(compute_type)
        # k[HEAD_SZ_POW2, KV_BLK_SZ_POW2]
        k = tl.permute(k, [1, 3, 0, 2]) # [HEAD_SZ_POW2/x, x, KV_BLK_PER_ITER, KV_BLK_SZ_POW2]
        k = tl.reshape(k, [HEAD_SZ_POW2, KV_BLK_PER_ITER * KV_BLK_SZ_POW2])

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_PER_ITER * KV_BLK_SZ_POW2]
        qk = tl.dot(q, k, out_dtype=tl.float32)

        blk_seq_offs_flatten = blk_seq_offs.reshape([KV_BLK_PER_ITER * KV_BLK_SZ_POW2])
        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (blk_seq_offs_flatten - seq_len + 1)[None, :]).to(
                tl.float32
            )
        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs_flatten[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        # max_logit_new: [QUERY_GRP_SZ_POW2]
        max_logit_new = tl.maximum(max_logit, tl.max(qk, axis=1))

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_PER_ITER * KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        p = p.to(compute_type)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        exp_sum = exp_sum * alpha + tl.sum(p, axis=1)
        max_logit = max_logit_new
        acc *= alpha[:, None]

        # load v[KV_BLK_PER_ITER, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
        v_blk_offs = kv_blk_nums[:, None, None] * stride_v_b + v_offs[None, :, :]
        v_mask = (
            (blk_seq_offs[:, None, :] < seq_len)   # blk_seq_offs: [KV_BLK_SZ_POW2]
            & (blk_offs[None, None, :] < KV_BLK_SZ)   # blk_offs: [KV_BLK_SZ_POW2]
            & (head_sz_offs[None, :, None] < HEAD_SZ) # head_sz_offs: [HEAD_SZ_POW2]
        )
        v_0 = tl.load(v_cache_ptr + v_blk_offs)
        v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
        v = v.to(compute_type)
        v = tl.permute(v, [0, 2, 1]) # [KV_BLK_PER_ITER, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
        v = tl.reshape(v, [KV_BLK_PER_ITER * KV_BLK_SZ_POW2, HEAD_SZ_POW2])

        acc += tl.dot(p, v, out_dtype=tl.float32)

    acc = acc / exp_sum[:, None]

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs, max_logit, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)

    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )

    tl.store(logits_ptr + logits_offs, acc, mask=q_mask)

@triton.jit
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

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    max_num_kv_blks: tl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    head_sz_div_offs = tl.arange(0, HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
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
    blk_ids = tl.arange(0, max_num_kv_blks)
    masked_blk_ids = tl.where(blk_ids < num_kv_blks, blk_ids, 0)
    kv_blk_start = seq_part_idx * max_num_kv_blks
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

    # k_blk_offs[max_num_kv_blks, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_blk_offs = (
        kv_blk_nums[:, None, None, None] * stride_k_b
        + kv_head_idx * stride_k_nh
        + head_sz_div_offs[None, :, None, None] * stride_k_hz
        + blk_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
        + contiguous_kv_elems_offs[None, None, None, :]
    )
    # blk_seq_offs[max_num_kv_blks, KV_BLK_SZ_POW2]
    blk_seq_offs = ((kv_blk_start + blk_ids[:, None]) * KV_BLK_SZ  # blk_ids: [max_num_kv_blks]
                    + blk_offs[None, :]) # blk_offs: [KV_BLK_SZ_POW2]

    k_mask = (
        (blk_seq_offs[:, None, :, None] < seq_len) &
        (blk_offs[None, None, :, None] < KV_BLK_SZ) &
        (head_sz_div_offs[None, :, None, None] < (HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD))
    )

    # k[max_num_kv_blks, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_0 = tl.load(k_cache_ptr + k_blk_offs)
    k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
    k = k.to(compute_type)
    # k[HEAD_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2]
    k = tl.permute(k, [1, 3, 0, 2]) # [HEAD_SZ_POW2/x, x, max_num_kv_blks, KV_BLK_SZ_POW2]
    k = tl.reshape(k, [HEAD_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2])

    # qk[QUERY_GRP_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2]
    qk = tl.dot(q, k, out_dtype=tl.float32)
    blk_seq_flatten_offs = tl.reshape(blk_seq_offs, [max_num_kv_blks * KV_BLK_SZ_POW2])
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
    # p[QUERY_GRP_SZ_POW2, max_num_kv_blks * KV_BLK_SZ_POW2]
    p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
    p = p.to(compute_type)
    exp_sum = tl.sum(p, axis=1)

    # v_blk_offs[max_num_kv_blks, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
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

    # v[max_num_kv_blks, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    v_0 = tl.load(v_cache_ptr + v_blk_offs)
    v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
    v = v.to(compute_type)
    # v[max_num_kv_blks * KV_BLK_SZ_POW2, HEAD_SZ_POW2]
    v = tl.permute(v, [0, 2, 1])
    v = tl.reshape(v, [max_num_kv_blks * KV_BLK_SZ_POW2, HEAD_SZ_POW2])

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


# Reshape KV cache before execution
# Suffer from k_cache loading without prefetching (150 us)
@triton.jit
def _paged_attn_decode_v2_w_dot_kernel_reshape_load4(
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
    stride_v_kb,
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
    LANE16_SIZE: tl.constexpr = 16
    NUM_WARPS: tl.constexpr = 4
    TOKENS_PER_ITER: tl.constexpr = LANE16_SIZE * NUM_WARPS
    KV_BLK_PER_ITER: tl.constexpr = (TOKENS_PER_ITER + KV_BLK_SZ - 1) // KV_BLK_SZ
    NUM_ITER: tl.constexpr = (SEQ_PARTITION_SZ + TOKENS_PER_ITER - 1) // TOKENS_PER_ITER
    MAX_NUM_KV_BLKS: tl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    MAX_SEQ_LEN = (tl.num_programs(2) * SEQ_PARTITION_SZ) // KV_BLK_SZ
    MAX_NUM_KV_BLKS_IN_SEQ = (MAX_SEQ_LEN + KV_BLK_SZ - 1) // KV_BLK_SZ

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if seq_part_idx * SEQ_PARTITION_SZ >= seq_len:
        return

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)
    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    head_sz_div_offs = tl.arange(0, HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    contiguous_kv_elems_offs = tl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD)
    kv_blk_per_iter_offs = tl.arange(0, KV_BLK_PER_ITER)
    tokens_per_iter_offs = tl.arange(0, TOKENS_PER_ITER)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)

    # k_offs[KV_BLK_PER_ITER, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_offs = (
        kv_head_idx * stride_v_nh
        + head_sz_div_offs[:, None, None] * stride_k_hz
        + blk_offs[None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
        + contiguous_kv_elems_offs[None, None, :]
    )
    # v_offs[HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    v_offs = (
        kv_head_idx * stride_v_nh
        + blk_offs[None, :]                     # blk_offs: [KV_BLK_SZ_POW2]
        + head_sz_offs[:, None] * stride_v_kb   # head_sz_offs: [HEAD_SZ_POW2]
    )
    kv_blk_start = seq_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    for iter in range(NUM_ITER):
        kv_blk_idx = iter * KV_BLK_PER_ITER + kv_blk_per_iter_offs
        kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_start + kv_blk_idx, mask=(kv_blk_idx < num_kv_blks), other=0)

        k_blk_offs = kv_blk_nums[:, None, None, None] * stride_k_b + k_offs[None, :, :, :]
        blk_seq_offs = kv_blk_idx[:, None] * KV_BLK_SZ + blk_offs[None, :]
        k_mask = (
            (blk_seq_offs[:, None, :, None] < seq_len)   # blk_seq_offs: [KV_BLK_SZ_POW2]
            & (blk_offs[None, None, :, None] < KV_BLK_SZ)   # blk_offs: [KV_BLK_SZ_POW2]
            & (head_sz_div_offs[None, :, None, None] < (HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD)) # head_sz_offs: [HEAD_SZ_POW2]
        )

        # load k[KV_BLK_PER_ITER, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
        k_0 = tl.load(k_cache_ptr + k_blk_offs)
        k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
        k = k.to(compute_type)
        # k[HEAD_SZ_POW2, KV_BLK_SZ_POW2]
        k = tl.permute(k, [0, 2, 1, 3]) # [KV_BLK_PER_ITER, KV_BLK_SZ_POW2, HEAD_SZ_POW2/x, x]
        k = tl.reshape(k, [KV_BLK_PER_ITER * KV_BLK_SZ_POW2, HEAD_SZ_POW2])

        # qk: [KV_BLK_PER_ITER * KV_BLK_SZ_POW2, QUERY_GRP_SZ_POW2]
        qk = tl.dot(k, q.T, out_dtype=tl.float32)

        blk_seq_offs_flatten = blk_seq_offs.reshape([KV_BLK_PER_ITER * KV_BLK_SZ_POW2])
        if alibi_slopes is not None:
            qk += (alibi_slope[None, :] * (blk_seq_offs_flatten - seq_len + 1)[:, None]).to(
                tl.float32
            )
        qk = tl.where(
            (q_grp_offs[None, :] < QUERY_GRP_SZ) & (blk_seq_offs_flatten[:, None] < seq_len),
            qk,
            float("-inf"),
        )

        # max_logit_new: [QUERY_GRP_SZ_POW2]
        max_logit_new = tl.maximum(max_logit, tl.max(qk, axis=0))

        # p: [KV_BLK_PER_ITER * KV_BLK_SZ_POW2, QUERY_GRP_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[None, :]) * log2e)
        p = p.to(compute_type)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        exp_sum = exp_sum * alpha + tl.sum(p, axis=0)
        max_logit = max_logit_new
        acc *= alpha[:, None]

        # load v[KV_BLK_PER_ITER, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
        v_blk_offs = kv_blk_nums[:, None, None] * stride_v_b + v_offs[None, :, :]
        v_mask = (
            (blk_seq_offs[:, None, :] < seq_len)      # blk_seq_offs: [KV_BLK_SZ_POW2]
            & (blk_offs[None, None, :] < KV_BLK_SZ)   # blk_offs: [KV_BLK_SZ_POW2]
            & (head_sz_offs[None, :, None] < HEAD_SZ) # head_sz_offs: [HEAD_SZ_POW2]
        )
        v_0 = tl.load(v_cache_ptr + v_blk_offs)
        v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
        v = v.to(compute_type)
        v = tl.permute(v, [1, 0, 2]) # [HEAD_SZ_POW2, KV_BLK_PER_ITER, KV_BLK_SZ_POW2]
        v = tl.reshape(v, [HEAD_SZ_POW2, KV_BLK_PER_ITER * KV_BLK_SZ_POW2])

        acc += tl.dot(v, p, out_dtype=tl.float32).T

    acc = acc / exp_sum[:, None]

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs, max_logit, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)

    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )

    tl.store(logits_ptr + logits_offs, acc, mask=q_mask)

@triton.jit
def _paged_attn_decode_v2_w_dot_reduce_kernel(
    out_ptr,        # [num_seqs, num_kv_heads, q_grp_sz, head_sz]
    exp_sums_ptr,   # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr, # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptrs,    # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    seq_lens_ptr,   # [num_seqs]
    stride_o_s,
    stride_o_h,
    stride_exp_sums_s,
    stride_exp_sums_h,
    stride_exp_sums_p,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_logits_g,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_partitions = tl.cdiv(seq_len, SEQ_PARTITION_SZ)

    part_offs = tl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    head_offs = tl.arange(0, HEAD_SZ_POW2)

    # get global max logit
    exp_sums_offs = (
        seq_idx * stride_exp_sums_s
        + kv_head_idx * stride_exp_sums_h
        + part_offs[:, None] * stride_exp_sums_p
        + q_grp_offs[None, :]
    )
    exp_sums_mask = (part_offs[:, None] < num_partitions) & (
        q_grp_offs[None, :] < QUERY_GRP_SZ
    )

    # max_logits: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    max_logits = tl.load(
        max_logits_ptr + exp_sums_offs, mask=exp_sums_mask, other=float("-inf")
    )
    # max_logit: [QUERY_GRP_SZ_POW2]
    ml = tl.max(max_logits, axis=0)

    # Rescale the exp sums and compute the global sum
    # exp_sums: [MAX_NUM_SEQ_PARTITIONS, QUERY_GRP_SZ_POW2]
    exp_sums = tl.load(exp_sums_ptr + exp_sums_offs, mask=exp_sums_mask, other=0.0)
    exp_sums *= tl.exp(max_logits - ml[None, :])

    # exp_sum: [QUERY_GRP_SZ_POW2]
    exp_sum = tl.sum(exp_sums, axis=0)

    # p: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    p = exp_sums / exp_sum[None, :]
    p = tl.reshape(p, (MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2, 1))

    # logits_offset
    logits_offset = (
        seq_idx * stride_logits_s
        + kv_head_idx * stride_logits_h
        + part_offs[:, None, None] * stride_logits_p
        + q_grp_offs[None, :, None] * stride_logits_g
        + head_offs[None, None, :]
    )
    # load logits
    logits_mask = (part_offs[:, None] < num_partitions) & (
        q_grp_offs[None, :] < QUERY_GRP_SZ
    )
    logits = tl.load(
        logits_ptrs + logits_offset, mask=logits_mask[:, :, None], other=0.0
    )

    # out: [QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    out = tl.sum((logits * p).to(tl.float32), axis=0).to(out_ptr.dtype.element_ty)

    # store output
    out_offs = (
        seq_idx * stride_o_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_o_h
        + head_offs[None, :]
    )
    tl.store(
        out_ptr + out_offs,
        out,
        mask=(q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_offs[None, :] < HEAD_SZ),
    )


def paged_attn_decode_v1_per_token_quant(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = query.shape[1] // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA- Multi-Head Attention
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v1_wo_dot_kernel_per_token_quant[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            query.stride(0),
            query.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            MAX_SEQ_LEN_POW2=max_seq_len,
        )
    # GQA - Grouped Query Attention
    else:
        grid = (num_seqs, num_kv_heads, 1)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v1_w_dot_kernel_per_token_quant[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz,
        )


@triton.jit
def _paged_attn_decode_v1_wo_dot_kernel_per_token_quant(
    out,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    q_ptr,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    blk_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_q_heads]
    scale,
    k_scale_ptr,  # [num_blks, num_kv_heads, kv_blk_sz]
    v_scale_ptr,  # [num_blks, num_kv_heads, kv_blk_sz]
    stride_q_s,
    stride_q_h,
    stride_o_s,
    stride_o_nh,
    stride_o_hs,
    stride_k_b,
    stride_k_nh,
    stride_k_kb,
    stride_bt_s,
    stride_k_scale_b,
    stride_k_scale_nh,
    stride_k_scale_kb,
    compute_type: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    MAX_SEQ_LEN_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    head_idx = tl.program_id(axis=0)
    seq_idx = tl.program_id(axis=1)
    kv_head_idx = head_idx // QUERY_GRP_SZ

    log2e: tl.constexpr = 1.4426950408889634

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    num_kv_blks = tl.cdiv(seq_len, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)

    # load alibi slopes [1]
    if alibi_slopes_ptr is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes_ptr + head_idx)

    # load q [1, HEAD_SZ_POW2]
    q_offs = seq_idx * stride_q_s + head_idx * stride_q_h + head_sz_offs
    q = tl.load(q_ptr + q_offs, mask=head_sz_offs < HEAD_SZ)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([KV_BLK_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = float("-inf")
    exp_sum = 0.0

    kv_offs = (
        kv_head_idx * stride_k_nh
        + blk_offs[:, None] * stride_k_kb
        + head_sz_offs[None, :]
    )
    k_scale_offs = kv_head_idx * stride_k_scale_nh + blk_offs * stride_k_scale_kb
    blk_tbl_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_nums = tl.load(blk_tbl_start_ptr + b)
        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = b * KV_BLK_SZ + blk_offs
        kv_mask = (
            (blk_seq_offs[:, None] < seq_len)
            & (blk_offs[:, None] < KV_BLK_SZ)
            & (head_sz_offs[None, :] < HEAD_SZ)
        )
        kv_scale_mask = (blk_seq_offs < seq_len) & (blk_offs < KV_BLK_SZ)
        kv_scale_offs = kv_blk_nums * stride_k_scale_b + k_scale_offs

        # load k [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_scale = tl.load(k_scale_ptr + kv_scale_offs, mask=kv_scale_mask, other=0.0)
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        if k_0.dtype.is_fp8():
            k = k_0.to(tl.float32) * k_scale[:, None]
        else:
            k = k_0
        k = k.to(compute_type)

        # qk #[KV_BLK_SZ_POW2]
        qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))
        if alibi_slopes_ptr is not None:
            qk += (alibi_slope * (blk_seq_offs - seq_len + 1)).to(tl.float32)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        max_logit_new = tl.maximum(tl.max(qk, axis=0), max_logit)

        # p: [KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # load v [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v_scale = tl.load(v_scale_ptr + kv_scale_offs, mask=kv_scale_mask, other=0.0)
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask)
        if v_0.dtype.is_fp8():
            v = v_0.to(tl.float32) * v_scale[:, None]
        else:
            v = v_0
        v = v.to(compute_type)

        acc += p[:, None] * v

        exp_sum = exp_sum * alpha + tl.sum(p, axis=0)
        max_logit = max_logit_new

    acc = acc / exp_sum

    offs_out = seq_idx * stride_o_s + head_idx * stride_o_nh + head_sz_offs
    out_mask = head_sz_offs < HEAD_SZ
    tl.store(
        out + offs_out, tl.sum(acc, axis=0).to(out.dtype.element_ty), mask=out_mask
    )


@triton.jit
def _paged_attn_decode_v1_w_dot_kernel_per_token_quant(
    out_ptr,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    q_ptr,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,  # [num_blocks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr,  # [num_blocks, num_kv_heads, kv_blk_sz, head_sz]
    blk_tables_ptr,  # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes,  # [num_kv_heads*query_grp_sz]
    scale,
    k_scale_ptr,  # [num_blks, num_kv_heads, kv_blk_sz]
    v_scale_ptr,  # [num_blks, num_kv_heads, kv_blk_sz]
    stride_o_s,
    stride_o_nh,
    stride_o_hs,
    stride_q_s,
    stride_q_nh,
    stride_q_hs,
    stride_k_b,
    stride_k_nh,
    stride_k_kb,
    stride_k_hs,
    stride_bt_s,
    stride_bt_nb,
    stride_k_scale_b,
    stride_k_scale_nh,
    stride_k_scale_kb,
    compute_type: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    log2e: tl.constexpr = 1.4426950408889634

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    num_kv_blks = tl.cdiv(seq_len, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :] * stride_q_hs
    )

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)

    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)

    kv_offs = (
        kv_head_idx * stride_k_nh
        + blk_offs[:, None] * stride_k_kb
        + head_sz_offs[None, :] * stride_k_hs
    )
    k_scale_offs = kv_head_idx * stride_k_scale_nh + blk_offs * stride_k_scale_kb
    blk_tbl_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_nums = tl.load(blk_tbl_start_ptr + b)
        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = b * KV_BLK_SZ + blk_offs
        kv_mask = (
            (blk_seq_offs[:, None] < seq_len)
            & (blk_offs[:, None] < KV_BLK_SZ)
            & (head_sz_offs[None, :] < HEAD_SZ)
        )

        kv_scale_mask = (blk_seq_offs < seq_len) & (blk_offs < KV_BLK_SZ)
        kv_scale_offs = kv_blk_nums * stride_k_scale_b + k_scale_offs

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_scale = tl.load(k_scale_ptr + kv_scale_offs, mask=kv_scale_mask, other=0.0)
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale[:, None] if k_0.dtype.is_fp8() else k_0
        k = k.to(compute_type)

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (blk_seq_offs - seq_len + 1)[None, :]).to(
                tl.float32
            )

        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )
        max_logit_new = tl.maximum(tl.max(qk, axis=1), max_logit)

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ, HEAD_SZ]
        v_scale = tl.load(v_scale_ptr + kv_scale_offs, mask=kv_scale_mask, other=0.0)
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        v = v_0.to(tl.float32) * v_scale[:, None] if v_0.dtype.is_fp8() else v_0
        v = v.to(compute_type)

        p = p.to(v.dtype)
        acc += tl.dot(p, v, out_dtype=tl.float32)

        exp_sum = exp_sum * alpha + tl.sum(p, axis=1)
        max_logit = max_logit_new

    acc = acc / exp_sum[:, None]

    out_offs = (
        seq_idx * stride_o_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_o_nh
        + head_sz_offs[None, :]
    )

    out_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def paged_attn_decode_v2_per_token_quant(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq],
    seq_lens: torch.Tensor,  # [num_seqs],
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    max_num_partitions: int,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)

    # Note: There is a bug in triton.next_power_of_2 function which causes it
    # to update the passed in arg, so that's why we have a workaround here
    # max_num_partitions_pow2 = triton.next_power_of_2(max_num_partitions)
    if max_num_partitions == 0:
        max_num_partitions_pow2 = 1
    else:
        max_num_partitions_pow2 = 2 ** math.ceil(math.log2(max_num_partitions))

    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, max_num_partitions)
        shape_info = (num_seqs, num_q_heads, max_num_partitions)
        exp_sums = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        max_logits = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        tmp_output = torch.empty(
            (*shape_info, head_sz), dtype=output.dtype, device=output.device
        )
        _paged_attn_decode_v2_wo_dot_kernel_per_token_quant[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            block_tables.stride(1),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_BLKS_PER_SEQ=block_tables.shape[1],
            MAX_SEQ_LEN_POW2=max_seq_len,
        )
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v2_wo_dot_reduce_kernel_per_token_quant[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS=int(max_num_partitions),
            MAX_NUM_SEQ_PARTITIONS_POW2=int(max_num_partitions_pow2),
        )
    # GQA
    else:
        grid = (num_seqs, num_kv_heads, max_num_partitions)
        shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
        max_logits = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        exp_sums = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        tmp_output = torch.empty(
            *shape_info, head_sz, dtype=output.dtype, device=output.device
        )
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v2_w_dot_kernel_per_token_quant[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        )
        grid = (num_seqs, num_kv_heads, 1)
        _paged_attn_decode_v2_w_dot_reduce_kernel_per_token_quant[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS=int(max_num_partitions),
            MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions)),
        )


@triton.jit
def _paged_attn_decode_v2_wo_dot_kernel_per_token_quant(
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    blk_tables_ptr,
    seq_lens_ptr,
    scale,
    k_scale_ptr,
    v_scale_ptr,
    alibi_slopes,
    stride_exp_s,
    stride_exp_h,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_q_s,
    stride_q_h,
    stride_k_b,
    stride_k_nh,
    stride_k_kb,
    stride_bt_s,
    stride_bt_nb,
    stride_k_scale_b,
    stride_k_scale_nh,
    stride_k_scale_kb,
    compute_type: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_BLKS_PER_SEQ: tl.constexpr,
    MAX_SEQ_LEN_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    seq_part_idx = tl.program_id(2)
    kv_head_idx = head_idx // QUERY_GRP_SZ

    log2e: tl.constexpr = 1.4426950408889634

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if seq_part_idx * SEQ_PARTITION_SZ >= seq_len:
        return

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)

    # load alibi slopes
    if alibi_slopes is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes + head_idx)

    # load q[HEAD_SZ]
    q_offs = seq_idx * stride_q_s + head_idx * stride_q_h + head_sz_offs
    q = tl.load(q_ptr + q_offs, mask=head_sz_offs < HEAD_SZ)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([KV_BLK_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = float("-inf")
    exp_sum = 0.0

    kv_offs = (
        kv_head_idx * stride_k_nh
        + blk_offs[:, None] * stride_k_kb
        + head_sz_offs[None, :]
    )
    kv_blk_start = seq_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    k_scale_offs = kv_head_idx * stride_k_scale_nh + blk_offs * stride_k_scale_kb
    blk_tables_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_idx = kv_blk_start + b
        kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_idx * stride_bt_nb)

        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = kv_blk_idx * KV_BLK_SZ + blk_offs
        kv_mask = (
            (blk_seq_offs[:, None] < seq_len)
            & (blk_offs[:, None] < KV_BLK_SZ)
            & (head_sz_offs[None, :] < HEAD_SZ)
        )

        kv_scale_mask = (blk_seq_offs < seq_len) & (blk_offs < KV_BLK_SZ)
        kv_scale_offs = kv_blk_nums * stride_k_scale_b + k_scale_offs

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_scale = tl.load(k_scale_ptr + kv_scale_offs, mask=kv_scale_mask, other=0.0)
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale[:, None] if k_0.dtype.is_fp8() else k_0
        k = k.to(compute_type)

        # qk: [KV_BLK_SZ_POW2]
        qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        if alibi_slopes is not None:
            qk += (alibi_slope * (blk_seq_offs - seq_len + 1)).to(tl.float32)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        max_logit_new = tl.maximum(max_logit, tl.max(qk, axis=0))

        # p: [KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v_scale = tl.load(v_scale_ptr + kv_scale_offs, mask=kv_scale_mask, other=0.0)
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        v = v_0.to(tl.float32) * v_scale[:, None] if v_0.dtype.is_fp8() else v_0
        v = v.to(compute_type)

        # acc: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        acc += p[:, None] * v

        exp_sum = exp_sum * alpha + tl.sum(p, axis=0)
        max_logit = max_logit_new

    acc = acc / exp_sum

    max_logits_offs = seq_idx * stride_exp_s + head_idx * stride_exp_h + seq_part_idx

    tl.store(max_logits_ptr + max_logits_offs, max_logit)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum)

    logits_offs = (
        seq_idx * stride_logits_s
        + head_idx * stride_logits_h
        + seq_part_idx * stride_logits_p
        + head_sz_offs
    )
    logits_mask = head_sz_offs < HEAD_SZ
    tl.store(
        logits_ptr + logits_offs,
        tl.sum(acc, axis=0).to(logits_ptr.dtype.element_ty),
        mask=logits_mask,
    )


@triton.jit
def _paged_attn_decode_v2_wo_dot_reduce_kernel_per_token_quant(
    out,
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptr,
    seq_lens,
    stride_out_n,
    stride_out_h,
    stride_exp_sums_n,
    stride_exp_sums_h,
    stride_logits_n,
    stride_logits_h,
    stride_logits_b,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    # get seq_idx, head_idx, seq_len
    head_idx = tl.program_id(axis=0)
    seq_idx = tl.program_id(axis=1)

    seq_len = tl.load(seq_lens + seq_idx)
    num_partitions = tl.cdiv(seq_len, SEQ_PARTITION_SZ)

    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    seq_part_offs = tl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2)

    max_logit = float("-inf")
    acc = tl.zeros([HEAD_SZ_POW2], dtype=tl.float32)
    global_exp_sum = tl.zeros([1], dtype=tl.float32)

    # load max_logits [MAX_NUM_SEQ_PARTITIONS_POW2]
    max_logits_offs = (
        seq_idx * stride_exp_sums_n + head_idx * stride_exp_sums_h + seq_part_offs
    )
    max_logits_mask = seq_part_offs < num_partitions
    max_logits = tl.load(
        max_logits_ptr + max_logits_offs,
        mask=max_logits_mask,
        other=float("-inf"),
    )

    # find max_logit
    max_logit = tl.max(max_logits, axis=0)

    # load exp_sum [MAX_NUM_SEQ_PARTITIONS_POW2]
    exp_sums_offs = (
        seq_idx * stride_exp_sums_n + head_idx * stride_exp_sums_h + seq_part_offs
    )
    exp_sums_mask = seq_part_offs < num_partitions
    exp_sums = tl.load(
        exp_sums_ptr + exp_sums_offs,
        mask=exp_sums_mask,
        other=0.0,
    )

    # rescaled_exp_sum and global_exp_sum
    # [MAX_NUM_SEQ_PARTITIONS_POW2]
    rescaled_exp_sum = exp_sums * tl.exp(max_logits - max_logit)
    global_exp_sum += tl.sum(rescaled_exp_sum, axis=0)
    rescaled_exp_sum /= global_exp_sum

    # load logits
    logits_offs = (
        seq_idx * stride_logits_n
        + head_idx * stride_logits_h
        + seq_part_offs[:, None] * stride_logits_b
        + head_sz_offs[None, :]
    )
    logits_mask = (seq_part_offs[:, None] < num_partitions) & (
        head_sz_offs[None, :] < HEAD_SZ
    )

    logits = tl.load(logits_ptr + logits_offs, mask=logits_mask, other=0.0)
    acc += tl.sum(logits * rescaled_exp_sum[:, None], axis=0)

    # store the final output
    out_ptr = seq_idx * stride_out_n + head_idx * stride_out_h + head_sz_offs
    out_mask = head_sz_offs < HEAD_SZ
    tl.store(out + out_ptr, acc.to(out.dtype.element_ty), mask=out_mask)


@triton.jit
def _paged_attn_decode_v2_w_dot_kernel_per_token_quant(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    q_ptr,  # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    blk_tables_ptrs,  # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale,
    k_scale_ptr,  # [num_blks, num_kv_heads, kv_blk_sz]
    v_scale_ptr,  # [num_blks, num_kv_heads, kv_blk_sz]
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
    stride_k_kb,
    stride_bt_s,
    stride_k_scale_b,
    stride_k_scale_nh,
    stride_k_scale_kb,
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

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if seq_part_idx * SEQ_PARTITION_SZ >= seq_len:
        return

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)

    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(compute_type)

    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)

    kv_offs = (
        kv_head_idx * stride_k_nh
        + blk_offs[:, None] * stride_k_kb
        + head_sz_offs[None, :]
    )
    kv_blk_start = seq_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    k_scale_offs = kv_head_idx * stride_k_scale_nh + blk_offs * stride_k_scale_kb
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    for b in range(num_kv_blks):
        kv_blk_idx = kv_blk_start + b
        kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_idx)

        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = kv_blk_idx * KV_BLK_SZ + blk_offs
        kv_mask = (
            (blk_seq_offs[:, None] < seq_len)
            & (blk_offs[:, None] < KV_BLK_SZ)
            & (head_sz_offs[None, :] < HEAD_SZ)
        )

        kv_scale_mask = (blk_seq_offs < seq_len) & (blk_offs < KV_BLK_SZ)
        kv_scale_offs = kv_blk_nums * stride_k_scale_b + k_scale_offs

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_scale = tl.load(k_scale_ptr + kv_scale_offs, mask=kv_scale_mask, other=0.0)
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale[:, None] if k_0.dtype.is_fp8() else k_0
        k = k.to(compute_type)

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (blk_seq_offs - seq_len + 1)[None, :]).to(
                tl.float32
            )
        qk = tl.where(
            (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        max_logit_new = tl.maximum(max_logit, tl.max(qk, axis=1))

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v_scale = tl.load(v_scale_ptr + kv_scale_offs, mask=kv_scale_mask, other=0.0)
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        v = v_0.to(tl.float32) * v_scale[:, None] if v_0.dtype.is_fp8() else v_0
        v = v.to(compute_type)

        p = p.to(v.dtype)
        acc += tl.dot(p, v, out_dtype=tl.float32)

        exp_sum = exp_sum * alpha + tl.sum(p, axis=1)
        max_logit = max_logit_new

    acc = acc / exp_sum[:, None]

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs, max_logit, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)

    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )

    tl.store(logits_ptr + logits_offs, acc, mask=q_mask)


@triton.jit
def _paged_attn_decode_v2_w_dot_reduce_kernel_per_token_quant(
    out_ptr,  # [num_seqs, num_kv_heads, q_grp_sz, head_sz]
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptrs,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    seq_lens_ptr,  # [num_seqs]
    stride_o_s,
    stride_o_h,
    stride_exp_sums_s,
    stride_exp_sums_h,
    stride_exp_sums_p,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_logits_g,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_partitions = tl.cdiv(seq_len, SEQ_PARTITION_SZ)

    part_offs = tl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    head_offs = tl.arange(0, HEAD_SZ_POW2)

    # get global max logit
    exp_sums_offs = (
        seq_idx * stride_exp_sums_s
        + kv_head_idx * stride_exp_sums_h
        + part_offs[:, None] * stride_exp_sums_p
        + q_grp_offs[None, :]
    )
    exp_sums_mask = (part_offs[:, None] < num_partitions) & (
        q_grp_offs[None, :] < QUERY_GRP_SZ
    )

    # max_logits: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    max_logits = tl.load(
        max_logits_ptr + exp_sums_offs, mask=exp_sums_mask, other=float("-inf")
    )
    # max_logit: [QUERY_GRP_SZ_POW2]
    ml = tl.max(max_logits, axis=0)

    # Rescale the exp sums and compute the global sum
    # exp_sums: [MAX_NUM_SEQ_PARTITIONS, QUERY_GRP_SZ_POW2]
    exp_sums = tl.load(exp_sums_ptr + exp_sums_offs, mask=exp_sums_mask, other=0.0)
    exp_sums *= tl.exp(max_logits - ml[None, :])

    # exp_sum: [QUERY_GRP_SZ_POW2]
    exp_sum = tl.sum(exp_sums, axis=0)

    # p: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    p = exp_sums / exp_sum[None, :]
    p = tl.reshape(p, (MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2, 1))

    # logits_offset
    logits_offset = (
        seq_idx * stride_logits_s
        + kv_head_idx * stride_logits_h
        + part_offs[:, None, None] * stride_logits_p
        + q_grp_offs[None, :, None] * stride_logits_g
        + head_offs[None, None, :]
    )
    # load logits
    logits_mask = (part_offs[:, None] < num_partitions) & (
        q_grp_offs[None, :] < QUERY_GRP_SZ
    )
    logits = tl.load(
        logits_ptrs + logits_offset, mask=logits_mask[:, :, None], other=0.0
    )

    # out: [QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    out = tl.sum((logits * p).to(tl.float32), axis=0).to(out_ptr.dtype.element_ty)

    # store output
    out_offs = (
        seq_idx * stride_o_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_o_h
        + head_offs[None, :]
    )
    tl.store(
        out_ptr + out_offs,
        out,
        mask=(q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_offs[None, :] < HEAD_SZ),
    )
