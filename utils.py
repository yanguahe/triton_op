from typing import List, Dict
import numpy as np


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

    # print("\nTop differences:")
    # for item in result['top_k_diff']:
    #     print(f"Position {item['position']}: arr1 = {arr1[item['position']]:.6f}, arr2 = {arr2[item['position']]:.6f}, Diff = {item['value']:.6f}")

    # print("\nThreshold statistics:")
    # for stat in result['threshold_stats']:
    #     print(f"{stat['range']}: {stat['count']} ({stat['percentage']:.2f}%)")

    print("\nNaN info:")
    print(result['nan_info'])

    return result
