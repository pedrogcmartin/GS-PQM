#!/usr/bin/env python3


import argparse
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import json
import os
from openpyxl import Workbook, load_workbook
from skimage.color import rgb2yuv

PARAM_GROUPS = {
    "position": ["x", "y", "z"],
    #"rotation": ["rot_0", "rot_1", "rot_2", "rot_3"],
    #"scale": ["scale_0", "scale_1", "scale_2"],
    #"opacity": ["opacity"],
    "sh_dc": ["f_dc_0", "f_dc_1", "f_dc_2"],
    "sh_ac": [f"f_rest_{i}" for i in range(0, 45)],
    #"sh_ac_d1": ["f_rest_0", "f_rest_1", "f_rest_2", "f_rest_15", "f_rest_16", "f_rest_17", "f_rest_30", "f_rest_31", "f_rest_32"],
    #"sh_ac_d2": ["f_rest_3", "f_rest_4", "f_rest_5", "f_rest_6", "f_rest_7", "f_rest_18", "f_rest_19", "f_rest_20", "f_rest_21", "f_rest_22", "f_rest_33", "f_rest_34", "f_rest_35", "f_rest_36", "f_rest_37"],
    #"sh_ac_d3": ["f_rest_8", "f_rest_9", "f_rest_10", "f_rest_11", "f_rest_12", "f_rest_13", "f_rest_14", "f_rest_23", "f_rest_24", "f_rest_25", "f_rest_26", "f_rest_27", "f_rest_28", "f_rest_29", "f_rest_38", "f_rest_39", "f_rest_40", "f_rest_41", "f_rest_42", "f_rest_43", "f_rest_44"],
    #"sh_ac_d1": [f"f_rest_{i}" for i in range(0, 9)],
    #"sh_ac_d2": [f"f_rest_{i}" for i in range(9, 24)],
    #"sh_ac_d3": [f"f_rest_{i}" for i in range(24, 45)],
    "sh_ac_1": ["f_rest_0", "f_rest_15", "f_rest_30"],
    "sh_ac_2": ["f_rest_1", "f_rest_16", "f_rest_31"],
    "sh_ac_3": ["f_rest_2", "f_rest_17", "f_rest_32"],
    "sh_ac_4": ["f_rest_3", "f_rest_18", "f_rest_33"],
    "sh_ac_5": ["f_rest_4", "f_rest_19", "f_rest_34"],
    "sh_ac_6": ["f_rest_5", "f_rest_20", "f_rest_35"],
    "sh_ac_7": ["f_rest_6", "f_rest_21", "f_rest_36"],
    "sh_ac_8": ["f_rest_7", "f_rest_22", "f_rest_37"],
    "sh_ac_9": ["f_rest_8", "f_rest_23", "f_rest_38"],
    "sh_ac_10": ["f_rest_9", "f_rest_24", "f_rest_39"],
    "sh_ac_11": ["f_rest_10", "f_rest_25", "f_rest_40"],
    "sh_ac_12": ["f_rest_11", "f_rest_26", "f_rest_41"],
    "sh_ac_13": ["f_rest_12", "f_rest_27", "f_rest_42"],
    "sh_ac_14": ["f_rest_13", "f_rest_28", "f_rest_43"],
    "sh_ac_15": ["f_rest_14", "f_rest_29", "f_rest_44"],
}

ALL_SH_AC_FIELDS = [f"f_rest_{i}" for i in range(45)]


def read_ply_params(path):
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"{path}: no vertex element found")

    data = ply["vertex"].data
    names = data.dtype.names

    out = {}
    for name in names:
        out[name] = np.asarray(data[name], dtype=np.float64)

    return out


def stack_params(params, fields, path, fill_missing_zero=False):
    n = len(next(iter(params.values())))

    values = []
    missing = []

    for f in fields:
        if f in params:
            values.append(params[f])
        else:
            missing.append(f)
            if fill_missing_zero:
                values.append(np.zeros(n, dtype=np.float64))
            else:
                raise ValueError(f"{path}: missing fields {missing}")

    if missing:
        print(f"[INFO] {path}: missing fields filled with zero: {missing}")

    return np.stack(values, axis=1)

def finite_mask(arr):
    return np.all(np.isfinite(arr), axis=1)

def nearest_neighbor_indices(src_xyz, dst_xyz):
    tree = cKDTree(dst_xyz)
    _, idx = tree.query(src_xyz, k=1)
    return idx

def nearest_neighbor_distances_and_indices(src_xyz, dst_xyz):
    """Return Euclidean nearest-neighbour distances and indices."""
    if len(dst_xyz) == 0:
        raise ValueError("The destination model contains no valid Gaussians.")
    tree = cKDTree(dst_xyz)
    distances, indices = tree.query(src_xyz, k=1, workers=-1)
    return distances, indices


def required_parameter_fields():
    """Return unique PLY fields needed by the enabled parameter groups."""
    fields = []
    seen = set()
    for group_fields in PARAM_GROUPS.values():
        for field in group_fields:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def radius_average_target_parameters(
    src_xyz,
    dst_xyz,
    dst_params,
    dst_path,
    sigma=1.0,
    chunk_size=10000,
    verbose=False,
):
    """
    For every source Gaussian:
      1. Find nearest-target distance d_i.
      2. Set radius r_i = sigma * d_i.
      3. Find all target Gaussians inside r_i.
      4. Average their stored parameter values.

    The returned averaged parameter dictionary contains one synthetic average
    target Gaussian for every source Gaussian.
    """
    if sigma < 1.0:
        raise ValueError(
            f"sigma must be at least 1.0 so the nearest Gaussian is included, "
            f"got {sigma}."
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}.")
    if len(src_xyz) == 0 or len(dst_xyz) == 0:
        raise ValueError("Source and destination models must be non-empty.")

    fields = required_parameter_fields()
    dst_matrix = stack_params(
        dst_params,
        fields,
        dst_path,
        fill_missing_zero=True,
    )

    dst_tree = cKDTree(dst_xyz)
    nearest_distances, nearest_indices = dst_tree.query(
        src_xyz, k=1, workers=-1
    )
    radii = np.nextafter(sigma * nearest_distances, np.inf)

    n_src = len(src_xyz)
    averaged_matrix = np.empty(
        (n_src, len(fields)),
        dtype=np.float64,
    )
    candidate_counts = np.empty(n_src, dtype=np.int64)

    for start in range(0, n_src, chunk_size):
        end = min(start + chunk_size, n_src)

        candidate_lists = dst_tree.query_ball_point(
            src_xyz[start:end],
            r=radii[start:end],
            workers=-1,
        )

        for local_i, candidate_list in enumerate(candidate_lists):
            source_i = start + local_i

            if len(candidate_list) == 0:
                candidate_idx = np.asarray(
                    [nearest_indices[source_i]],
                    dtype=np.int64,
                )
            else:
                candidate_idx = np.asarray(
                    candidate_list,
                    dtype=np.int64,
                )

            candidate_counts[source_i] = len(candidate_idx)
            averaged_matrix[source_i] = np.mean(
                dst_matrix[candidate_idx],
                axis=0,
                dtype=np.float64,
            )

    averaged_params = {
        field: averaged_matrix[:, column]
        for column, field in enumerate(fields)
    }

    if verbose:
        print(f"[RADIUS AVERAGING] alpha={sigma:.6g}")
        print(
            "  Candidate count: "
            f"mean={np.mean(candidate_counts):.3f}, "
            f"median={np.median(candidate_counts):.3f}, "
            f"min={np.min(candidate_counts)}, "
            f"max={np.max(candidate_counts)}"
        )

    return averaged_params, nearest_distances, radii, candidate_counts


def compute_gaussian_volume(scale_params):
    return np.exp(scale_params[:, 0]) * np.exp(scale_params[:, 1]) * np.exp(scale_params[:, 2])

def mse_sh_dc_yuv(sh_ref, sh_dist):
    C0 = 0.28209479177387814
    
    # 1. Standard 3DGS DC to RGB conversion
    rgb_ref = np.clip(0.5 + C0 * sh_ref, 0.0, 1.0)
    rgb_dist = np.clip(0.5 + C0 * sh_dist, 0.0, 1.0)
    
    # 2. Map RGB into YUV space
    yuv_ref = rgb2yuv(rgb_ref)
    yuv_dist = rgb2yuv(rgb_dist)
    
    # 3. Return squared Euclidean distance across channels
    return np.mean(np.sum((yuv_ref - yuv_dist) ** 2, axis=1))


def mse_sh_dc_y_and_uv(sh_ref, sh_dist):
    C0 = 0.28209479177387814

    rgb_ref = np.clip(0.5 + C0 * sh_ref, 0.0, 1.0)
    rgb_dist = np.clip(0.5 + C0 * sh_dist, 0.0, 1.0)

    yuv_ref = rgb2yuv(rgb_ref)
    yuv_dist = rgb2yuv(rgb_dist)

    mse_y = np.mean((yuv_ref[:, 0] - yuv_dist[:, 0]) ** 2)
    mse_uv = np.mean(np.sum((yuv_ref[:, 1:3] - yuv_dist[:, 1:3]) ** 2, axis=1))

    return mse_y, mse_uv

def mse_sh_dc_yuv_411(sh_ref, sh_dist):
    C0 = 0.28209479177387814

    weights = np.array([1.0, 0.25, 0.25])  # YUV 4:1:1

    rgb_ref = np.clip(0.5 + C0 * sh_ref, 0.0, 1.0)
    rgb_dist = np.clip(0.5 + C0 * sh_dist, 0.0, 1.0)

    yuv_ref = rgb2yuv(rgb_ref)
    yuv_dist = rgb2yuv(rgb_dist)

    diff = yuv_ref - yuv_dist

    return np.mean(np.sum(weights * diff**2, axis=1))

def mse_sh_ac_yuv(sh_ref, sh_dist):
    # sh_ref / sh_dist have shape (N, channels) where channels = 9, 15, or 21
    # Determine the number of SH coefficients per color channel (3 channels total: R, G, B)
    num_coeffs = sh_ref.shape[1] // 3
    
    # Reshape matching your script's conversion logic: (N, num_coeffs, 3) using Fortran order
    ref_reshaped = sh_ref.reshape(-1, num_coeffs, 3, order='F')
    dist_reshaped = sh_dist.reshape(-1, num_coeffs, 3, order='F')
    
    # Convert every individual coefficient slice across all Gaussians
    ref_yuv = np.zeros_like(ref_reshaped)
    dist_yuv = np.zeros_like(dist_reshaped)
    
    for i in range(num_coeffs):
        ref_yuv[:, i, :] = rgb2yuv(ref_reshaped[:, i, :])
        dist_yuv[:, i, :] = rgb2yuv(dist_reshaped[:, i, :])
        
    # Return Euclidean distance across the transformed channels
    return np.mean(np.sum((ref_yuv - dist_yuv) ** 2, axis=(1, 2)))

def mse_sh_ac_yuv_411(sh_ref, sh_dist):
    num_coeffs = sh_ref.shape[1] // 3

    weights = np.array([1.0, 0.25, 0.25])  # YUV 4:1:1

    ref_reshaped = sh_ref.reshape(-1, num_coeffs, 3, order='F')
    dist_reshaped = sh_dist.reshape(-1, num_coeffs, 3, order='F')

    ref_yuv = np.zeros_like(ref_reshaped)
    dist_yuv = np.zeros_like(dist_reshaped)

    for i in range(num_coeffs):
        ref_yuv[:, i, :] = rgb2yuv(ref_reshaped[:, i, :])
        dist_yuv[:, i, :] = rgb2yuv(dist_reshaped[:, i, :])

    diff = ref_yuv - dist_yuv

    return np.mean(np.sum(weights * diff**2, axis=(1, 2)))

def mse_sh_ac_y_and_uv(sh_ref, sh_dist):
    num_coeffs = sh_ref.shape[1] // 3

    """print("\nReference before\n")
    print(sh_ref)
    print("\nTest before\n")
    print(sh_dist)"""

    ref_reshaped = sh_ref.reshape(-1, num_coeffs, 3, order="F")
    dist_reshaped = sh_dist.reshape(-1, num_coeffs, 3, order="F")

    ref_yuv = np.zeros_like(ref_reshaped)
    dist_yuv = np.zeros_like(dist_reshaped)

    for i in range(num_coeffs):
        ref_yuv[:, i, :] = rgb2yuv(ref_reshaped[:, i, :])
        dist_yuv[:, i, :] = rgb2yuv(dist_reshaped[:, i, :])

    """print("\nReference after\n")
    print(ref_yuv[:, :, 1:3])
    print("\nTest after\n")
    print(dist_yuv[:, :, 1:3])"""

    mse_y = np.mean(np.sum((ref_yuv[:, :, 0] - dist_yuv[:, :, 0]) ** 2, axis=1))
    mse_uv = np.mean(np.sum((ref_yuv[:, :, 1:3] - dist_yuv[:, :, 1:3]) ** 2, axis=(1, 2)))

    return mse_y, mse_uv

def mse_vector(a, b):
    return np.mean(np.sum((a - b) ** 2, axis=1))

def mse_componentwise(a, b):
    return np.mean((a - b) ** 2)

def mse_opacity_sigmoid(ref_raw, dist_raw):
    # 1. Apply sigmoid to both reference and distorted values
    alpha_ref = 1.0 / (1.0 + np.exp(-ref_raw))
    alpha_dist = 1.0 / (1.0 + np.exp(-dist_raw))
    
    # 2. Compute the mean squared error in alpha space [0, 1]
    return np.mean((alpha_ref - alpha_dist) ** 2)

def normalize_quaternions(q):
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    # Prevent division by zero for any corrupt/zero vertices
    norm = np.where(norm == 0, 1.0, norm)
    return q / norm

def mse_rotation_quaternion(q_ref, q_dist):
    q_ref_n = normalize_quaternions(q_ref)
    q_dist_n = normalize_quaternions(q_dist)
    
    d1 = np.sum((q_ref_n - q_dist_n) ** 2, axis=1)
    d2 = np.sum((q_ref_n + q_dist_n) ** 2, axis=1)
    return np.mean(np.minimum(d1, d2))

def mse_rotation_quaternion_angular(q_ref, q_dist):
    q_ref_n = normalize_quaternions(q_ref)
    q_dist_n = normalize_quaternions(q_dist)
    
    # 1. Compute the dot product of normalized vectors
    dot_product = np.sum(q_ref_n * q_dist_n, axis=1)
    
    # 2. Clamp values to safely handle floating-point precision limits
    dot_product = np.clip(np.abs(dot_product), 0.0, 1.0)
    
    # 3. True angular distance theta = 2 * acos(|dot|)
    angles = 2.0 * np.arccos(dot_product)
    
    return np.mean(angles ** 2)

def mse_scale_linear(s_ref, s_dist):
    # Convert log-scales to physical linear dimensions (a, b, c)
    s_ref_linear = np.exp(s_ref)
    s_dist_linear = np.exp(s_dist)
    
    # Compute standard Euclidean MSE across the 3 linear scale dimensions
    return np.mean((s_ref_linear - s_dist_linear) ** 2)

def compute_group_metrics(group_name, fields, ref_params, dist_params, avg_dist_for_ref, avg_ref_for_dist, ref_path, dist_path):
    ref_values = stack_params(
        ref_params, fields, ref_path, fill_missing_zero=False
    )
    dist_values = stack_params(
        dist_params, fields, dist_path, fill_missing_zero=True
    )
    avg_dist_values = stack_params(
        avg_dist_for_ref,
        fields,
        "averaged distorted neighbourhoods",
        fill_missing_zero=True,
    )
    avg_ref_values = stack_params(
        avg_ref_for_dist,
        fields,
        "averaged reference neighbourhoods",
        fill_missing_zero=True,
    )

    ref_to_dist_ref = ref_values
    ref_to_dist_dist = avg_dist_values

    dist_to_ref_dist = dist_values
    dist_to_ref_ref = avg_ref_values

    outputs = []

    if group_name.startswith("sh_"):
        # 1. Compute Raw Metric (as originally done)
        mse_fwd = mse_vector(ref_to_dist_ref, ref_to_dist_dist)
        mse_bwd = mse_vector(dist_to_ref_ref, dist_to_ref_dist)
        mse_sym = max(mse_fwd, mse_bwd)
        outputs.append((f"{group_name}", mse_fwd, mse_bwd, mse_sym))
        is_dc = (group_name == "sh_dc")

        # 3. Compute YUV Metric
        if is_dc:
            # Base color tracking using the activated RGB -> YUV method
            mse_fwd_yuv = mse_sh_dc_yuv(ref_to_dist_ref, ref_to_dist_dist)
            mse_bwd_yuv = mse_sh_dc_yuv(dist_to_ref_ref, dist_to_ref_dist)
            mse_fwd_yuv_411 = mse_sh_dc_yuv_411(ref_to_dist_ref, ref_to_dist_dist)
            mse_bwd_yuv_411 = mse_sh_dc_yuv_411(dist_to_ref_ref, dist_to_ref_dist)
            mse_fwd_y, mse_fwd_uv = mse_sh_dc_y_and_uv(ref_to_dist_ref, ref_to_dist_dist)
            mse_bwd_y, mse_bwd_uv = mse_sh_dc_y_and_uv(dist_to_ref_ref, dist_to_ref_dist)
        
        else:
            # AC band tracking using the structural Fortran reshape loop
            mse_fwd_yuv = mse_sh_ac_yuv(ref_to_dist_ref, ref_to_dist_dist)
            mse_bwd_yuv = mse_sh_ac_yuv(dist_to_ref_ref, dist_to_ref_dist)
            mse_fwd_yuv_411 = mse_sh_ac_yuv_411(ref_to_dist_ref, ref_to_dist_dist)
            mse_bwd_yuv_411 = mse_sh_ac_yuv_411(dist_to_ref_ref, dist_to_ref_dist)
            mse_fwd_y, mse_fwd_uv = mse_sh_ac_y_and_uv(ref_to_dist_ref, ref_to_dist_dist)
            mse_bwd_y, mse_bwd_uv = mse_sh_ac_y_and_uv(dist_to_ref_ref, dist_to_ref_dist)
        
        mse_sym_yuv = max(mse_fwd_yuv, mse_bwd_yuv)
        mse_sym_y = max(mse_fwd_y, mse_bwd_y)
        mse_sym_uv = max(mse_fwd_uv, mse_bwd_uv)
        mse_sym_yuv_411 = max(mse_fwd_yuv_411, mse_bwd_yuv_411)
        outputs.append((f"{group_name}_yuv", mse_fwd_yuv, mse_bwd_yuv, mse_sym_yuv))
        outputs.append((f"{group_name}_y", mse_fwd_y, mse_bwd_y, mse_sym_y))
        outputs.append((f"{group_name}_uv", mse_fwd_uv, mse_bwd_uv, mse_sym_uv))
        outputs.append((f"{group_name}_yuv_411", mse_fwd_yuv_411, mse_bwd_yuv_411, mse_sym_yuv_411))

    elif group_name == "opacity":
        # Metric 1: Raw Opacity (As stored in the PLY)
        mse_fwd = mse_vector(ref_to_dist_ref, ref_to_dist_dist)
        mse_bwd = mse_vector(dist_to_ref_ref, dist_to_ref_dist)
        mse_sym = max(mse_fwd, mse_bwd)
        outputs.append(("opacity", mse_fwd, mse_bwd, mse_sym))

        # Metric 2: Sigmoid Opacity (Perceptual/Rendering impact)
        mse_fwd_s = mse_opacity_sigmoid(ref_to_dist_ref, ref_to_dist_dist)
        mse_bwd_s = mse_opacity_sigmoid(dist_to_ref_ref, dist_to_ref_dist)
        mse_sym_s = max(mse_fwd_s, mse_bwd_s)
        
        # The peak for activated opacity is always 1.0 (range from 0.0 to 1.0)
        outputs.append(("opacity_sigmoid", mse_fwd_s, mse_bwd_s, mse_sym_s))

    elif group_name == "scale":
        # Metric 1: Raw Scale Parameters (Original)
        mse_fwd = mse_vector(ref_to_dist_ref, ref_to_dist_dist)
        mse_bwd = mse_vector(dist_to_ref_ref, dist_to_ref_dist)
        mse_sym = max(mse_fwd, mse_bwd)
        outputs.append(("scale", mse_fwd, mse_bwd, mse_sym))

        # Metric 2: Real Scale Parameters
        mse_fwd_lin = mse_scale_linear(ref_to_dist_ref, ref_to_dist_dist)
        mse_bwd_lin = mse_scale_linear(dist_to_ref_ref, dist_to_ref_dist)
        mse_sym_lin = max(mse_fwd_lin, mse_bwd_lin)
        ref_linear = np.exp(ref_values)
        outputs.append(("scale_linear", mse_fwd_lin, mse_bwd_lin, mse_sym_lin))

        # Metric 3: Gaussian Volume
        vol_ref = compute_gaussian_volume(ref_values)
        vol_dist = compute_gaussian_volume(dist_values)
        
        # Align volumes spatially
        ref_to_dist_vol_dist = vol_dist[idx_ref_to_dist]
        dist_to_ref_vol_ref = vol_ref[idx_dist_to_ref]

        # Component-wise MSE (dim=1)
        mse_fwd_v = np.mean((vol_ref - ref_to_dist_vol_dist) ** 2)
        mse_bwd_v = np.mean((vol_dist - dist_to_ref_vol_ref) ** 2)
        mse_sym_v = max(mse_fwd_v, mse_bwd_v)
        outputs.append(("volume", mse_fwd_v, mse_bwd_v, mse_sym_v))

    elif group_name == "rotation":
        # Metric 1: Chordal (The original method)
        mse_fwd = mse_rotation_quaternion(ref_to_dist_ref, ref_to_dist_dist)
        mse_bwd = mse_rotation_quaternion(dist_to_ref_ref, dist_to_ref_dist)
        mse_sym = max(mse_fwd, mse_bwd)
        outputs.append(("rotation", mse_fwd, mse_bwd, mse_sym))

        # Metric 2: Geodesic (The angular method)
        mse_fwd_g = mse_rotation_quaternion_angular(ref_to_dist_ref, ref_to_dist_dist)
        mse_bwd_g = mse_rotation_quaternion_angular(dist_to_ref_ref, dist_to_ref_dist)
        mse_sym_g = max(mse_fwd_g, mse_bwd_g)
        outputs.append(("rotation_geodesic", mse_fwd_g, mse_bwd_g, mse_sym_g))
    else:
        # Standard processing for all other groups
        mse_fwd = mse_vector(ref_to_dist_ref, ref_to_dist_dist)
        mse_bwd = mse_vector(dist_to_ref_ref, dist_to_ref_dist)
        mse_sym = max(mse_fwd, mse_bwd)
        outputs.append((group_name, mse_fwd, mse_bwd, mse_sym))

    return outputs


def compute_gs_dist(ref_path, dist_path, alpha=5.0, chunk_size=10000, verbose=False):
    """
    Compute the four GS-Dist errors used by GS-PQM.

    Parameters
    ----------
    ref_path : str
        Path to the uncompressed reference GS PLY file.

    dist_path : str
        Path to the compressed/distorted GS PLY file.

    alpha : float
        Radius multiplier used to define the adaptive neighbourhood.

    chunk_size : int
        Number of source Gaussians processed per radius-search chunk.

    Returns
    -------
    dict
        Four symmetric GS-Dist errors used by GS-PQM.
    """

    ref_params = read_ply_params(ref_path)
    dist_params = read_ply_params(dist_path)

    ref_xyz = stack_params(
        ref_params, ["x", "y", "z"], ref_path
    )

    dist_xyz = stack_params(
        dist_params, ["x", "y", "z"], dist_path
    )

    # Remove Gaussians with invalid positions.
    ref_mask = finite_mask(ref_xyz)
    dist_mask = finite_mask(dist_xyz)

    ref_xyz = ref_xyz[ref_mask]
    dist_xyz = dist_xyz[dist_mask]

    for key in ref_params:
        ref_params[key] = ref_params[key][ref_mask]

    for key in dist_params:
        dist_params[key] = dist_params[key][dist_mask]

    # Reference -> distorted adaptive neighbourhoods.
    avg_dist_for_ref, _, _, _ = radius_average_target_parameters(
        src_xyz=ref_xyz,
        dst_xyz=dist_xyz,
        dst_params=dist_params,
        dst_path=dist_path,
        sigma=alpha,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    # Distorted -> reference adaptive neighbourhoods.
    avg_ref_for_dist, _, _, _ = radius_average_target_parameters(
        src_xyz=dist_xyz,
        dst_xyz=ref_xyz,
        dst_params=ref_params,
        dst_path=ref_path,
        sigma=alpha,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    results = {}

    # Only calculate the parameter groups required by GS-PQM.
    for group_name in ["position", "sh_dc", "sh_ac"]:

        group_results = compute_group_metrics(
            group_name=group_name,
            fields=PARAM_GROUPS[group_name],
            ref_params=ref_params,
            dist_params=dist_params,
            avg_dist_for_ref=avg_dist_for_ref,
            avg_ref_for_dist=avg_ref_for_dist,
            ref_path=ref_path,
            dist_path=dist_path,
        )

        for name, _, _, mse_sym in group_results:
            results[name] = float(mse_sym)

    required = [
        "position",
        "sh_dc_yuv_411",
        "sh_ac_y",
        "sh_ac_uv",
    ]

    missing = [name for name in required if name not in results]

    if missing:
        raise RuntimeError(
            f"Could not compute required GS-PQM errors: {missing}"
        )

    return {name: results[name] for name in required}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Reference GS .ply file")
    parser.add_argument("--dist", required=True, help="Distorted/compressed GS .ply file")
    parser.add_argument("--out-json", default="results/results.json", help="Path to save results as JSON")
    parser.add_argument("--out-xlsx", default="results/results.xlsx")
    parser.add_argument("--only", choices=list(PARAM_GROUPS.keys()), default=None, help="Compute only one parameter class")
    parser.add_argument("--sigma", type=float, default=5, required=True, help="Radius multiplier: r_i = sigma times the nearest-neighbour distance.",)
    parser.add_argument("--match-chunk-size", type=int, default=10000, help="Number of source Gaussians processed per radius-averaging chunk")

    args = parser.parse_args()

    ref_params = read_ply_params(args.ref)
    dist_params = read_ply_params(args.dist)

    ref_xyz = stack_params(ref_params, ["x", "y", "z"], args.ref)
    dist_xyz = stack_params(dist_params, ["x", "y", "z"], args.dist)

    ref_mask = finite_mask(ref_xyz)
    dist_mask = finite_mask(dist_xyz)

    ref_xyz = ref_xyz[ref_mask]
    dist_xyz = dist_xyz[dist_mask]

    for k in ref_params:
        ref_params[k] = ref_params[k][ref_mask]

    for k in dist_params:
        dist_params[k] = dist_params[k][dist_mask]

    (avg_dist_for_ref, nn_ref_to_dist, radius_ref_to_dist, count_ref_to_dist) = radius_average_target_parameters(src_xyz=ref_xyz, dst_xyz=dist_xyz, dst_params=dist_params, dst_path=args.dist, sigma=args.sigma, chunk_size=args.match_chunk_size)
    (avg_ref_for_dist, nn_dist_to_ref, radius_dist_to_ref, count_dist_to_ref) = radius_average_target_parameters(src_xyz=dist_xyz, dst_xyz=ref_xyz, dst_params=ref_params, dst_path=args.ref, sigma=args.sigma, chunk_size=args.match_chunk_size)

    groups = PARAM_GROUPS
    if args.only is not None:
        groups = {args.only: PARAM_GROUPS[args.only]}

    print("\nGS parameter-domain IV-MSE using Gaussian-to-average radius neighbourhoods\n")
    print(f"Reference: {args.ref}")
    print(f"Distorted: {args.dist}")

    results = {
        "reference": args.ref,
        "distorted": args.dist,
        "sigma": float(args.sigma),
        "radius_search": {
            "ref_to_dist_candidate_count_mean": float(np.mean(count_ref_to_dist)),
            "ref_to_dist_candidate_count_median": float(np.median(count_ref_to_dist)),
            "dist_to_ref_candidate_count_mean": float(np.mean(count_dist_to_ref)),
            "dist_to_ref_candidate_count_median": float(np.median(count_dist_to_ref)),
        },
        "metrics": {}
    }
    
    base_info = {
        "Reference": args.ref,
        "Distorted": args.dist,
        "Sigma": float(args.sigma),
        "Ref2Dist_candidates_mean": float(np.mean(count_ref_to_dist)),
        "Dist2Ref_candidates_mean": float(np.mean(count_dist_to_ref)),
    }
    excel_rows = {
        "MSE_ref2dist": base_info.copy(),
        "MSE_dist2ref": base_info.copy(),
        "MSE_sym": base_info.copy()
    }

    for group_name, fields in groups.items():
        try:
            # Get the list of results (will be 2 for rotation, 1 for others)
            group_results = compute_group_metrics(group_name=group_name, fields=fields, ref_params=ref_params, dist_params=dist_params, avg_dist_for_ref=avg_dist_for_ref, avg_ref_for_dist=avg_ref_for_dist, ref_path=args.ref, dist_path=args.dist)

            for name, mse_fwd, mse_bwd, mse_sym in group_results:
                print(f"[{name}]")
                print(f"  MSE ref→ dist:{mse_fwd:.12e}")
                print(f"  MSE dist→ ref:{mse_bwd:.12e}")
                print(f"  MSE sym:      {mse_sym:.12e}")

                results["metrics"][name] = {
                    "mse_ref_to_dist": float(mse_fwd),
                    "mse_dist_to_ref": float(mse_bwd),
                    "mse_symmetric": float(mse_sym)
                }

                excel_rows["MSE_ref2dist"][name] = float(mse_fwd)
                excel_rows["MSE_dist2ref"][name] = float(mse_bwd)
                excel_rows["MSE_sym"][name] = float(mse_sym)

        except ValueError as e:
            print(f"[{group_name}] skipped: {e}\n")

    # Load existing file if it exists
    if os.path.exists(args.out_json):
        with open(args.out_json, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # Ensure structure
    if "experiments" not in data:
        data["experiments"] = []

    # Append new experiment
    data["experiments"].append(results)

    out_json_dir = os.path.dirname(args.out_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(data, f, indent=4)

    # Save back
    base_path, ext = os.path.splitext(args.out_xlsx)
    if not ext:
        ext = ".xlsx"
    
    # Loop over the metrics, constructing unique file paths based on your requested names
    for metric_key, metrics_row in excel_rows.items():
        specific_xlsx_path = f"{base_path}_{metric_key}{ext}"
        
        if not os.path.exists(specific_xlsx_path):
            wb = Workbook()
            ws = wb.active
            columns = list(metrics_row.keys())
            ws.append(columns)
        else:
            wb = load_workbook(specific_xlsx_path)
            ws = wb.active
            columns = [cell.value for cell in ws[1]]
            
            for key in metrics_row.keys():
                if key not in columns:
                    columns.append(key)
                    ws.cell(row=1, column=len(columns)).value = key

        row_to_append = [metrics_row.get(col, "") for col in columns]
        ws.append(row_to_append)
        
        xlsx_dir = os.path.dirname(specific_xlsx_path)
        if xlsx_dir:
            os.makedirs(xlsx_dir, exist_ok=True)
        wb.save(specific_xlsx_path)
        print(f"Updated Excel output for {metric_key}: {specific_xlsx_path}")

if __name__ == "__main__":
    main()