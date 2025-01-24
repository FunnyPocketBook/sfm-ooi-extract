import numpy as np
import open3d as o3d
import os
import json
from scipy.spatial import cKDTree

def compute_metrics(gt_pcd, gen_pcd):
    """
    Computes various metrics to evaluate the accuracy of a generated point cloud against the ground truth.
    Handles cases where the point counts differ.
    
    Args:
        gt_points (numpy.ndarray): Ground truth points, shape (N1, 3).
        gen_points (numpy.ndarray): Generated points, shape (N2, 3).
        
    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Convert to numpy arrays
    gt_points = np.asarray(gt_pcd.points)
    gen_points = np.asarray(gen_pcd.points)

    # Compute pairwise distances using k-D tree for efficiency
    gt_kdtree = cKDTree(gt_points)
    gen_kdtree = cKDTree(gen_points)


    gt_set = set(map(tuple, gt_points))
    gen_set = set(map(tuple, gen_points))
    
    # For Chamfer Distance and Hausdorff Distance
    gen_to_gt_distances, _ = gen_kdtree.query(gt_points, k=1)
    gt_to_gen_distances, _ = gt_kdtree.query(gen_points, k=1)

    
    aabb = gt_pcd.get_axis_aligned_bounding_box()
    aabb_extent = aabb.get_extent()  # [dx, dy, dz]
    aabb_diagonal = np.linalg.norm(aabb_extent)  # Diagonal length
    
    # Metrics
    # Chamfer Distance (First Term)
    chamfer_distance = np.mean(gen_to_gt_distances ** 2) + np.mean(gt_to_gen_distances ** 2)
    
    # Hausdorff Distance
    hausdorff_distance = max(np.max(gen_to_gt_distances), np.max(gt_to_gen_distances))
    hausdorff_bb_ratio = hausdorff_distance / aabb_diagonal

    # MSE and RMSE
    mse = np.mean(gen_to_gt_distances ** 2)  # Only for nearest points
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(gen_to_gt_distances)  # Absolute distance
    
    # Signal-to-Noise Ratio (SNR)
    signal_power = np.sum(np.linalg.norm(gt_points, axis=1) ** 2)
    noise_power = np.sum(gen_to_gt_distances ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    # Exact matches
    exact_matches = gt_set & gen_set
    matched_count = len(exact_matches)
    
    # Missing and extra points
    missing_count = len(gt_set - gen_set)
    extra_count = len(gen_set - gt_set)
    
    # Completeness (Recall)
    completeness = matched_count / len(gt_set) if len(gt_set) > 0 else 1.0
    
    # Accuracy (Precision)
    accuracy = matched_count / len(gen_set) if len(gen_set) > 0 else 1.0
    
    # F1 Score
    if completeness + accuracy > 0:
        f1_score = 2 * (completeness * accuracy) / (completeness + accuracy)
    else:
        f1_score = 0.0
    
        
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Chamfer Distance": chamfer_distance,
        "Hausdorff Distance": hausdorff_distance,
        "Bounding Box Diagonal": aabb_diagonal,
        "Hausdorff / BB Ratio": hausdorff_bb_ratio,
        "SNR": snr,
        "Exact Matches": matched_count,
        "Missing Points": missing_count,
        "Extra Points": extra_count,
        "Completeness/Recall": completeness,
        "Accuracy/Precision": accuracy,
        "F1 Score": f1_score,
    }


def ply_metrics(gt_path, gen_dir, name, output_file):
    """
    Computes metrics for all .ply files in the given directories matching a specific naming pattern.
    
    Args:
        gt_dir (str): Directory containing the ground truth point clouds.
        gen_dir (str): Directory containing the generated point clouds.
        name (str): Base name for the point cloud files (e.g., "example").
        output_file (str): Path to the file where results will be saved.
        
    Returns:
        None
    """
    # Define the file patterns to process
    file_patterns = [f"{name}_dense.ply", f"{name}_final.ply", f"{name}_floorseg.ply"]
    results = {}
    gt_pcd = o3d.io.read_point_cloud(gt_path)

    for file_pattern in file_patterns:
        gen_path = os.path.join(gen_dir, file_pattern)
        print()
        print(gen_path)
        
        # Check if both files exist
        if not os.path.exists(gen_path):
            print(f"Skipping {file_pattern} as gen file is missing.")
            continue
        
        # Load point clouds
        gen_pcd = o3d.io.read_point_cloud(gen_path)
        
        # Convert to numpy arrays
        
        # Compute metrics
        metrics = compute_metrics(gt_pcd, gen_pcd)
        results[file_pattern] = metrics
        print(f"Metrics for {file_pattern}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Save results to a file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
    return results