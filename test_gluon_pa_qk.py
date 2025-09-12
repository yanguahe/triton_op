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


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, 
                   k: int = 5, 
                   thresholds: List[float] = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]) -> Dict:
    """
    Compare two numpy arrays and compute various difference metrics.
    
    Args:
        arr1: First input array (float32)
        arr2: Second input array (float32)
        k: Number of top differences to return
        thresholds: List of thresholds for difference magnitude analysis
        
    Returns:
        Dictionary containing:
        - top_k_diff: Top k absolute differences with their positions
        - threshold_stats: Count and percentage of differences above each threshold
        - nan_info: Information about NaN values in input arrays
    """
    # Check input shapes
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")
    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)

    result = {
        'top_k_diff': [],
        'threshold_stats': [],
        'nan_info': {}
    }

    # Check for NaN values
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    
    if np.any(nan_mask1):
        result['nan_info']['arr1_nan_count'] = np.sum(nan_mask1)
        result['nan_info']['arr1_nan_positions'] = np.argwhere(nan_mask1)
        print(f"Warning: arr1 contains {result['nan_info']['arr1_nan_count']} NaN values")
    
    if np.any(nan_mask2):
        result['nan_info']['arr2_nan_count'] = np.sum(nan_mask2)
        result['nan_info']['arr2_nan_positions'] = np.argwhere(nan_mask2)
        print(f"Warning: arr2 contains {result['nan_info']['arr2_nan_count']} NaN values")
    
    # Compute absolute differences
    diff = np.abs(arr1 - arr2)
    total_elements = arr1.size

    max_diff_thr = diff / (1.0 + np.abs(arr2))
    max_diff_thr = max_diff_thr.max()
    print(f"diff.abs.max={diff.max()}")
    print(f"max_diff_thr={max_diff_thr}")

    # Find top k differences
    flat_diff = diff.flatten()
    top_k_indices = np.argpartition(flat_diff, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(-flat_diff[top_k_indices])]

    # Convert flat indices to multi-dimensional indices
    orig_indices = np.unravel_index(top_k_indices, diff.shape)
    for i in range(k):
        idx = tuple(dim[i] for dim in orig_indices)
        result['top_k_diff'].append({
            'value': diff[idx],
            'position': idx,
            'arr1_value': arr1[idx],
            'arr2_value': arr2[idx]
        })

    # Compute threshold statistics
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]
        mask = (diff >= lower) & (diff < upper)
        count = np.sum(mask)
        result['threshold_stats'].append({
            'range': f"[{lower:.1e}, {upper:.1e})",
            'count': count,
            'percentage': 100 * count / total_elements
        })
    
    # Handle values above the largest threshold
    mask = diff >= thresholds[-1]
    count = np.sum(mask)
    result['threshold_stats'].append({
        'range': f">={thresholds[-1]:.1e}",
        'count': count,
        'percentage': 100 * count / total_elements
    })

    print("\nTop differences:")
    for item in result['top_k_diff']:
        print(f"Position {item['position']}: arr1 = {arr1[item['position']]:.6f}, arr2 = {arr2[item['position']]:.6f}, Diff = {item['value']:.6f}")

    print("\nThreshold statistics:")
    for stat in result['threshold_stats']:
        print(f"{stat['range']}: {stat['count']} ({stat['percentage']:.2f}%)")

    print("\nNaN info:")
    print(result['nan_info'])

    return result


def get_autotune_config():
    sizes = [
        # {'QUERY_GRP_SZ': 16, 'SEQ_PARTITION_SZ': 256, 'SEQ_PARTITION_KV_BLK_NUM': 16, 'K_HD_SPLIT_NUM': 16, 'K_SPLIT_HEAD_SZ': 8, 'KV_BLK_SZ': 16, 'HEAD_SZ': 128},
        {'QUERY_GRP_SZ': 16, 'SEQ_PARTITION_SZ': 256, 'SEQ_PARTITION_KV_BLK_NUM': 16, 'K_HD_SPLIT_NUM': 16, 'K_SPLIT_HEAD_SZ': 1, 'KV_BLK_SZ': 16, 'HEAD_SZ': 128},
    ]
    return [triton.Config(s) for s in sizes]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@gluon.jit
def gemm_qk(
    q_ptr,
    k_ptr,
    qk_ptr,
    kv_len,
    q_stride0,
    q_stride1,
    q_stride2,
    q_stride3,
    q_stride4,
    k_stride0,
    k_stride1,
    k_stride2,
    k_stride3,
    k_stride4,
    k_stride5,
    k_stride6,
    qk_stride0,
    qk_stride1,
    qk_stride2,
    qk_stride3,
    qk_stride4,
    qk_stride5,
    QUERY_GRP_SZ: gl.constexpr,
    SEQ_PARTITION_SZ: gl.constexpr,
    SEQ_PARTITION_KV_BLK_NUM: gl.constexpr,
    K_HD_SPLIT_NUM: gl.constexpr,
    K_SPLIT_HEAD_SZ: gl.constexpr,
    KV_BLK_SZ: gl.constexpr,
    HEAD_SZ: gl.constexpr,
    ):
    # - Q: Matrix Q with shape (batch_size, num_kv_heads * QUERY_GRP_SZ, HEAD_SZ).
    """
    Key parameters:
    - Q: Matrix Q with shape (batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ).
    - K: Matrix K with shape (batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ).
    - QK: Matrix QK with shape (batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ).
    - kv_len = seq_partition_kv_num * SEQ_PARTITION_KV_BLK_NUM * KV_BLK_SZ
    - K_SPLIT_HEAD_SZ = 8
    """

    seq_len = kv_len
    batch_id = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    seq_part_idx = gl.program_id(2)
    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    # if seq_start_idx >= seq_len:
    #     return

    q_base_offset = batch_id * q_stride0 + kv_head_idx * q_stride1
    k_base_offset = batch_id * k_stride0 + seq_part_idx * k_stride1 + kv_head_idx * k_stride3
    qk_base_offset = batch_id * qk_stride0 + seq_part_idx * qk_stride1 + kv_head_idx * qk_stride3
    q_ptr = q_ptr + q_base_offset
    k_ptr = k_ptr + k_base_offset
    qk_ptr = qk_ptr + qk_base_offset
    # seq_end_idx = gl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    # max_num_kv_blks: gl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    # num_kv_blks = gl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    # 1 x QUERY_GRP_SZ x K_HD_SPLIT_NUM
    # 1 x 8(mdim) x 16(kdim)
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[1, 1, 1],
        threads_per_warp=[1, 8, 8],
        warps_per_cta   =[1, 2, 2],
        order           =[2, 1, 0],
    )
    # SEQ_PARTITION_KV_BLK_NUM x K_HD_SPLIT_NUM x KV_BLK_SZ
    # 16 x 16(kdim) x 16(ndim)
    blocked_k: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[4, 4, 1],
        threads_per_warp=[1, 4, 16],
        warps_per_cta   =[4, 1, 1],
        order           =[2, 1, 0],
    )

    # transposed: indicates the result tensor is transposed so that each thread holds consecutive elements
    # in the same row instead of column, which is good for chained dot and global write.
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=False, warps_per_cta=[4, 1, 1]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    q_dim_0_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, blocked_q))
    # q_dim_0_layout: gl.constexpr = gl.SliceLayout(1, blocked_q)
    q_dim_1_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, blocked_q))
    q_dim_2_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, blocked_q))

    k_dim_0_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, blocked_k))
    k_dim_1_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, blocked_k))
    k_dim_2_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, blocked_k))

    # offs_q_dim0 = gl.arange(0, SEQ_PARTITION_KV_BLK_NUM, layout=q_dim_0_layout)
    offs_q_dim0 = gl.arange(0, 1, layout=q_dim_0_layout)
    offs_q_dim1 = gl.arange(0, QUERY_GRP_SZ, layout=q_dim_1_layout)
    offs_q_dim2 = gl.arange(0, K_HD_SPLIT_NUM, layout=q_dim_2_layout)
    offs_k_dim0 = gl.arange(0, SEQ_PARTITION_KV_BLK_NUM, layout=k_dim_0_layout)
    offs_k_dim1 = gl.arange(0, K_HD_SPLIT_NUM, layout=k_dim_1_layout)
    offs_k_dim2 = gl.arange(0, KV_BLK_SZ, layout=k_dim_2_layout)
    offs_q = offs_q_dim0[:, None, None] * 0 + offs_q_dim1[None, :, None] * q_stride2 + offs_q_dim2[None, None, :] * q_stride3
    offs_k = offs_k_dim0[:, None, None] * k_stride2 + offs_k_dim1[None, :, None] * k_stride4 + offs_k_dim2[None, None, :] * k_stride5

    # accumulator = gl.zeros((SEQ_PARTITION_KV_BLK_NUM, QUERY_GRP_SZ, KV_BLK_SZ), dtype=gl.float32, layout=mfma_layout)
    # # for k in range(0, K_SPLIT_HEAD_SZ):
    # for k in range(0, 2):
    #     # cur_q_ptr = q_ptr + k * q_stride4
    #     # cur_k_ptr = k_ptr + k * k_stride6
    #     # a = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=offs_q, mask=offs_q_dim2[None, None, :] < K_HD_SPLIT_NUM)
    #     # b = gl.amd.cdna3.buffer_load(ptr=k_ptr, offsets=offs_k, mask=offs_k_dim2[None, None, :] < KV_BLK_SZ)
    #     a = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=offs_q)
    #     b = gl.amd.cdna3.buffer_load(ptr=k_ptr, offsets=offs_k)
    #     a_broadcasted = tl.broadcast_to(a, SEQ_PARTITION_KV_BLK_NUM, QUERY_GRP_SZ, K_HD_SPLIT_NUM)
    #     a1 = gl.convert_layout(a_broadcasted, layout=dot_a_layout)
    #     b1 = gl.convert_layout(b, layout=dot_b_layout)
    #     accumulator = gl.amd.cdna3.mfma(a1, b1, accumulator)
    #     q_ptr += q_stride4
    #     k_ptr += k_stride6


    acc0 = gl.zeros((SEQ_PARTITION_KV_BLK_NUM, QUERY_GRP_SZ, KV_BLK_SZ), dtype=gl.float32, layout=mfma_layout)
    acc1 = gl.zeros((SEQ_PARTITION_KV_BLK_NUM, QUERY_GRP_SZ, KV_BLK_SZ), dtype=gl.float32, layout=mfma_layout)
    a = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=offs_q)
    b = gl.amd.cdna3.buffer_load(ptr=k_ptr, offsets=offs_k)
    a_broadcasted = tl.broadcast_to(a, SEQ_PARTITION_KV_BLK_NUM, QUERY_GRP_SZ, K_HD_SPLIT_NUM)
    a1 = gl.convert_layout(a_broadcasted, layout=dot_a_layout)
    b1 = gl.convert_layout(b, layout=dot_b_layout)
    out0 = gl.amd.cdna3.mfma(a1, b1, acc0)
    q_ptr += q_stride4
    k_ptr += k_stride6

    a = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=offs_q)
    b = gl.amd.cdna3.buffer_load(ptr=k_ptr, offsets=offs_k)
    a_broadcasted = tl.broadcast_to(a, SEQ_PARTITION_KV_BLK_NUM, QUERY_GRP_SZ, K_HD_SPLIT_NUM)
    a1 = gl.convert_layout(a_broadcasted, layout=dot_a_layout)
    b1 = gl.convert_layout(b, layout=dot_b_layout)
    out1 = gl.amd.cdna3.mfma(a1, b1, acc1)
    q_ptr += q_stride4
    k_ptr += k_stride6
    accumulator = out0 + out1

    qk = accumulator.to(q_ptr.dtype.element_ty)
    offs_qk_dim0 = gl.arange(0, SEQ_PARTITION_KV_BLK_NUM, layout=gl.SliceLayout(1, gl.SliceLayout(2, mfma_layout)))
    offs_qk_dim1 = gl.arange(0, QUERY_GRP_SZ, layout=gl.SliceLayout(0, gl.SliceLayout(2, mfma_layout)))
    offs_qk_dim2 = gl.arange(0, KV_BLK_SZ, layout=gl.SliceLayout(0, gl.SliceLayout(1, mfma_layout)))
    offs_qk = offs_qk_dim0[:, None, None] * qk_stride2 + offs_qk_dim1[None, :, None] * qk_stride4 + offs_qk_dim2[None, None, :] * qk_stride5
    # qk_mask = seq_start_idx + offs_qk_dim0[:, None, None] * KV_BLK_SZ < seq_len
    # gl.amd.cdna3.buffer_store(stored_value=qk, ptr=qk_ptr, offsets=offs_qk, mask=qk_mask)
    gl.amd.cdna3.buffer_store(stored_value=qk, ptr=qk_ptr, offsets=offs_qk)


@perftest()
def run_gemm_qk(q, k, dtype):
    batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ = q.shape
    _, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, _, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ = k.shape
    kv_len = seq_partition_kv_num * SEQ_PARTITION_KV_BLK_NUM * KV_BLK_SZ
    # qk = torch.randn((batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ), device="cuda", dtype=dtype)
    qk = torch.empty((batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ), device="cuda", dtype=dtype)
    # print(f"q.stride(4)={q.stride(4)}")
    # print(f"k.stride(6)={k.stride(6)}")

    grid = (batch_size, num_kv_heads, seq_partition_kv_num)
    gemm_qk[grid](
        q,
        k,
        qk,
        kv_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k.stride(4),
        k.stride(5),
        k.stride(6),
        qk.stride(0),
        qk.stride(1),
        qk.stride(2),
        qk.stride(3),
        qk.stride(4),
        qk.stride(5),
    )
    return qk


def qk_gemm_ref(q, k):
    batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ = q.shape
    _, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, _, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ = k.shape

    # Transpose and reshape and K to align with Q's dimensions for matrix multiplication
    k_transposed = k.transpose(-2, -1)  # Swap last two dimensions
    k_transposed = k_transposed.contiguous()
    k_reshaped = k_transposed.reshape(batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ, KV_BLK_SZ)
    # Reshape Q for matrix multiplication
    q_reshaped = q.reshape(batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ)
    q_reshaped = q_reshaped.contiguous()
    # Perform matrix multiplication using einsum
    # Reduction happens over the last dimension of Q and first of k_transposed (K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ)
    qk = torch.einsum('bhgd,bsnhdt->bsnhgt', q_reshaped, k_reshaped)
    qk = qk.contiguous()
    # # Reshape to final output shape
    # qk = qk.reshape(batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ)
    return qk


def test_gemm_qk():
    # - Q: Matrix Q with shape (batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ).
    # - K: Matrix K with shape (batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ).
    # - QK: Matrix QK with shape (batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ).
    # shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
    # tmp_output = torch.empty(
    #     *shape_info, head_sz, dtype=output.dtype, device=output.device
    # )
    # - kv_len = seq_partition_kv_num * SEQ_PARTITION_KV_BLK_NUM * KV_BLK_SZ
    # - K_SPLIT_HEAD_SZ = 8
    # q_shape_list = [
    #     [80, 1, 16, 16, 8],
    # ]
    # k_shape_list = [
    #     [80, 16, 16, 1, 16, 16, 8],
    # ]
    q_shape_list = [
        [1, 1, 16, 16, 8],
    ]
    k_shape_list = [
        [1, 1, 16, 1, 16, 16, 8],
    ]
    batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ = q_shape_list[0]
    batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ = k_shape_list[0]
    q_len = 1
    kv_len = seq_partition_kv_num * SEQ_PARTITION_KV_BLK_NUM * KV_BLK_SZ
    head_dim_qk = K_HD_SPLIT_NUM * K_SPLIT_HEAD_SZ
    dtype = torch.float16

    q = torch.randn((batch_size, num_kv_heads, QUERY_GRP_SZ, K_HD_SPLIT_NUM, K_SPLIT_HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, K_HD_SPLIT_NUM, KV_BLK_SZ, K_SPLIT_HEAD_SZ), device="cuda", dtype=dtype)

    qk_gn, avg_t_us_gn = run_gemm_qk(q, k, dtype)

    q = q[:, :, :, :, 0:2]
    k = k[:, :, :, :, :, :, 0:2]
    q = q.to(torch.float32).contiguous()
    k = k.to(torch.float32).contiguous()
    head_dim_qk = K_HD_SPLIT_NUM * 2
    # qk_ref = qk_gemm_ref(q, k)
    q = q.reshape(1, QUERY_GRP_SZ, head_dim_qk).contiguous()
    q_broadcasted = q.repeat(SEQ_PARTITION_KV_BLK_NUM, 1, 1)
    print(f"k.shape={k.shape}, k.dtype={k.dtype}")
    k_transposed = k.transpose(-2, -1).contiguous()
    print(f"k_transposed.shape={k_transposed.shape}, k_transposed.dtype={k_transposed.dtype}")
    k_transposed = k_transposed.reshape(SEQ_PARTITION_KV_BLK_NUM, head_dim_qk, KV_BLK_SZ)
    print(f"k_transposed.shape={k_transposed.shape}, k_transposed.dtype={k_transposed.dtype}")
    print(f"k_transposed.stride()={k_transposed.stride()}")
    k_transposed = k_transposed.contiguous()
    print(f"k_transposed.stride()={k_transposed.stride()}")
    print(f"k_transposed.shape={k_transposed.shape}, k_transposed.dtype={k_transposed.dtype}")
    k_transposed = k_transposed.reshape(SEQ_PARTITION_KV_BLK_NUM, head_dim_qk, KV_BLK_SZ)
    qk_ref = torch.bmm(q_broadcasted, k_transposed)
    qk_ref = qk_ref.reshape(batch_size, seq_partition_kv_num, SEQ_PARTITION_KV_BLK_NUM, num_kv_heads, QUERY_GRP_SZ, KV_BLK_SZ)
    qk_ref = qk_ref.to(dtype)

    print(f"kv_len={kv_len}")
    print(f"head_dim_qk={head_dim_qk}")
    # print(f"q.shape={q.shape}, q.dtype={q.dtype}")
    # print(f"k.shape={k.shape}, k.dtype={k.dtype}")
    print(f"q_broadcasted.shape={q_broadcasted.shape}, q_broadcasted.dtype={q_broadcasted.dtype}")
    print(f"k_transposed.shape={k_transposed.shape}, k_transposed.dtype={k_transposed.dtype}")
    print(f"qk_gn.shape={qk_gn.shape}, qk_gn.dtype={qk_gn.dtype}")
    print(f"qk_ref.shape={qk_gn.shape}, qk_ref.dtype={qk_gn.dtype}")
    print(f"q_broadcasted.stride()={q_broadcasted.stride()}")
    print(f"k_transposed.stride()={k_transposed.stride()}")
    # print(f"q.stride()={q.stride()}")
    # print(f"k.stride()={k.stride()}")
    print(f"qk_gn.stride()={qk_gn.stride()}")
    print(f"qk_ref.stride()={qk_ref.stride()}")
    torch.testing.assert_close(qk_gn, qk_ref, rtol=1e-3, atol=1e-3)


    # TFLOPS = 2 * q_len * kv_len * head_dim_qk / (1e6 * avg_t_us_gn)
    # band_width = (q_len * head_dim_qk + kv_len * head_dim_qk) * (torch.finfo(dtype).bits // 8) / (1.024 ** 4 * 1e3 * avg_t_us_gn)

    # qk_gn_md5 = hashlib.md5(qk_gn.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    # qk_ref_md5 = hashlib.md5(qk_ref.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()

    # qk_gn = qk_gn.to(torch.float32)
    # compare_arrays(qk_gn.detach().cpu().numpy(), qk_ref.detach().cpu().numpy())

    # print(f"qk_gn_md5={qk_gn_md5}")
    # print(f"qk_ref_md5={qk_ref_md5}")
    # print(f"avg_t_us_gn={avg_t_us_gn:.3f} us, TFLOPS={TFLOPS:.1f} TFLOPS, band_width={band_width:.1f} GB/s")

    # torch.testing.assert_close(qk_gn, qk_ref, rtol=1e-3, atol=1e-3)
    # # torch.testing.assert_close(qk_gn, qk_ref, rtol=1e-3, atol=1e-5)
    # # torch.testing.assert_close(qk_gn, qk_ref)
    print("\033[92mPASSED\033[0m")


test_gemm_qk()
