# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
from typing import Optional

import triton
import triton.language as tl
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import compile
import torch
from aiter.test_common import perftest
import tempfile
import subprocess
import os


TEST_NUM_ITERS = 101

# This code is derived from sglang and FLASHNN projects
# https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py

_SEQ_PARTITION_SIZE = 256  # HIP


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
    exp_sum = tl.sum(p, axis=1)
    p = p.to(compute_type)

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


    # o_row_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    # o_col_offs = tl.arange(0, 256)
    # o_mask = (o_row_offs[:, None] < QUERY_GRP_SZ) & (kv_blk_start + o_col_offs[None, :] < seq_len)
    # logits_offs = seq_idx * stride_logits_s
    # logits_offs += kv_head_idx * stride_logits_nh
    # logits_offs += (
    #     seq_part_idx * stride_logits_p
    #     + o_row_offs[:, None] * stride_logits_g
    #     + o_col_offs[None, :]
    # )
    # tl.store(logits_ptr + logits_offs, p, mask=o_mask)
    # # tl.store(logits_ptr + logits_offs, qk.to(compute_type), mask=o_mask)
    # # tl.store(logits_ptr + logits_offs, qk.to(compute_type))


    # acc[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    acc = tl.dot(p, v, out_dtype=tl.float32)
    acc = acc / exp_sum[:, None]
    acc = acc.to(compute_type)

    # end up computation
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
        # ttgir_file_path = os.path.join(os.path.dirname(__file__), "./ttgir/pa_noloop.ttgir")
        ttgir_file_path = os.path.join(os.path.dirname(__file__), "./ttgir/pa_implicit_convert_use_v2.ttgir")
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
        _paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk[grid](
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
    batch_size = block_tables.shape[0]
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    qlen = num_seqs // batch_size

    max_num_partitions = int((max_seq_len + _SEQ_PARTITION_SIZE - 1) // _SEQ_PARTITION_SIZE)

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = value_cache.shape[3]
    head_sz = value_cache.shape[2]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz *= qlen
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

    # print(f"query.shape={query.shape}")
    # print(f"key_cache.shape={key_cache.shape}")
    # print(f"value_cache.shape={value_cache.shape}")
    # print(f"output.shape={output.shape}")
    # print(f"block_tables.shape={block_tables.shape}")
    # print(f"query.dtype={query.dtype}")
    # print(f"key_cache.dtype={key_cache.dtype}")
    # print(f"value_cache.dtype={value_cache.dtype}")
    # print(f"output.dtype={output.dtype}")
    # print(f"block_tables.dtype={block_tables.dtype}")
    # input_config = dict(
    #     qlen=qlen,
    #     kv_type=compute_type,
    #     compute_type=compute_type,
    #     HEAD_SZ=head_sz,
    #     HEAD_SZ_POW2=head_sz_pow2,
    #     QUERY_GRP_SZ=query_grp_sz,
    #     QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
    #     KV_BLK_SZ=kv_blk_sz,
    #     KV_BLK_SZ_POW2=kv_blk_sz_pow2,
    #     SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
    # )
    # print(input_config)

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
        attn_scale,
        k_scale.item(),
        v_scale.item(),
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
        kv_type=compute_type,
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

    # decode_time = 0
    # reduce_time = 0
    # print(f"triton:\n{tmp_output[0]}")

    return {'triton_decode': decode_time,
            'triton_reduce': reduce_time,
            'tmp_output': tmp_output,
            'exp_sums': exp_sums,
            'max_logits': max_logits,
            'triton': decode_time + reduce_time}
