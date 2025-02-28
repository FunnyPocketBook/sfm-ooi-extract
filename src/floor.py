import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import open3d as o3d
import concurrent
import plotly.graph_objs as go

def find_floor_plane(points, distance_threshold=0.02, min_floor_points=100):
    """Find the floor plane in a point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )
    

    inlier_mask = np.zeros(len(points), dtype=bool)
    inlier_mask[inliers] = True
    
    print(f"Number of inlier indices: {len(inliers)}")
    print(f"Number of True values in inliers: {np.sum(inliers)}")
    
    floor_points = points[inlier_mask]
    non_floor_points = points[~inlier_mask]
    
    print(f"Number of floor points: {len(floor_points)}")
    print(f"Number of non-floor points: {len(non_floor_points)}")
    
    if len(floor_points) < min_floor_points:
        print(f"Warning: Found only {len(floor_points)} floor points. Might be unreliable.")
    
    return floor_points, non_floor_points, plane_model

def find_optimal_threshold_floor(points, 
                           initial_threshold=0.005, 
                           max_threshold=0.1, 
                           iterations=50):
    """
    Automatically find optimal distance threshold for floor detection without predefined floor ratio bounds.
    The function iteratively adjusts the threshold and monitors the change in floor_ratio to determine when to stop.
    
    Parameters:
    -----------
    points : np.ndarray
        Input point cloud as a NumPy array of shape (N, 3).
    initial_threshold : float, default=0.02
        Starting distance threshold value.
    max_threshold : float, default=0.1
        Maximum allowed threshold.
    iterations : int, default=10
        Maximum number of iterations for searching the optimal threshold.
    
    Returns:
    --------
    best_threshold : float
        The optimal distance threshold found.
    best_ratio : float
        The ratio of floor points corresponding to the optimal threshold.
    stats : dict
        Dictionary containing statistics about the threshold search process.
    """
    total_points = len(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Calculate point cloud statistics
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    median_distance = np.median(distances)
    
    # Initialize threshold search
    threshold = initial_threshold
    best_threshold = threshold
    best_ratio = 0
    
    stats = {
        'iterations': [],
        'thresholds': [],
        'floor_ratios': [],
        'improvements': [],
        'median_distance': median_distance
    }
    
    for iteration in range(iterations):
        # Segment plane with current threshold
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        floor_ratio = len(inliers) / total_points
        
        # Store statistics
        stats['iterations'].append(iteration)
        stats['thresholds'].append(threshold)
        stats['floor_ratios'].append(floor_ratio)        

        if threshold >= max_threshold:
            print(f"Stopping search: Reached maximum threshold {max_threshold}")
            break

        threshold += (max_threshold - initial_threshold) / iterations
        threshold = min(threshold, max_threshold)
    
    smoothed_ratios = gaussian_filter1d(stats['floor_ratios'], sigma=3)
    second_derivative = np.gradient(np.gradient(smoothed_ratios))
    # best threshold is the one with the lowest second derivative
    best_threshold = stats['thresholds'][np.argmin(second_derivative)]
    best_ratio = stats['floor_ratios'][np.argmin(second_derivative)]
    

    # Collect final statistics
    stats['optimal_threshold'] = best_threshold
    stats['final_floor_ratio'] = best_ratio
    stats['median_point_distance'] = median_distance
    
    return best_threshold, best_ratio, stats


def find_floor_plane_auto(points, out_path, min_floor_points=100):
    """
    Enhanced floor detection with automatic threshold selection.
    """
    optimal_threshold, floor_ratio, stats = find_optimal_threshold_floor(points)
    
    # Create the figure
    fig = go.Figure()

    # Plot floor ratio evolution
    fig.add_trace(go.Scatter(
        x=stats['thresholds'],
        y=stats['floor_ratios'],
        name='Floor Ratio',
        mode='lines+markers',
        line=dict(color='#ff7f0e'),  # Orange
    ))

    # Smoothed floor ratio
    smoothed_floor_ratio = gaussian_filter1d(stats['floor_ratios'], sigma=3)
    fig.add_trace(go.Scatter(
        x=stats['thresholds'],
        y=smoothed_floor_ratio,
        name='Smoothed Floor Ratio',
        mode='lines',
        line=dict(color='#2ca02c'),  # Green
    ))

    # Plot second derivative of floor ratio
    second_derivative = np.gradient(np.gradient(smoothed_floor_ratio))
    fig.add_trace(go.Scatter(
        x=stats['thresholds'],
        y=second_derivative,
        name='2nd Derivative of Floor Ratio',
        mode='lines',
        line=dict(color='#800080', dash='dash'),  # purple
        yaxis='y2'
    ))

    # Update layout
    fig.update_layout(
        title='Threshold Search Evolution',
        xaxis_title='Threshold',
        yaxis=dict(
            title='Floor Ratio',
            showgrid=True,  # Ensure gridlines are visible
            gridcolor='lightgray',  # Gridline color
            gridwidth=0.5,  # Gridline width
        ),
        legend=dict(
            x=1.02,  # Slightly outside the graph on the right
            y=1.0,  # Aligns with the top of the graph
            xanchor='left',  # Anchors the legend's left side at x=1.02
            yanchor='top',  # Anchors the legend's top side at y=1.0
        ),
        yaxis2=dict(
            title='2nd Derivative',
            overlaying='y',
            side='right',
            showgrid=False,  # No gridlines
        ),
        xaxis=dict(
            showgrid=True,  # Align the grid with the Y-axis
            gridcolor='lightgray',
            gridwidth=0.5,
        )
    )

    # Save plot as an HTML file
    fig.write_html(f"{out_path}/plots/threshold_search_evolution.html")

    print(f"Found optimal threshold: {optimal_threshold:.4f}")
    print(f"Floor ratio: {floor_ratio:.2%}")
    print(f"Median point distance: {stats['median_point_distance']:.4f}")

    # Call the main function with the optimal threshold
    return find_floor_plane(points, distance_threshold=optimal_threshold, 
                            min_floor_points=min_floor_points)

def determine_model_orientation(points, plane_model):
    """Determine if the model is upside down relative to the floor plane."""
    print("Determining model orientation...")
    a, b, c, d = plane_model
    normal_vector = np.array([a, b, c])
    
    signed_distances = (points @ normal_vector + d)
    
    points_above = points[signed_distances > 0]
    points_below = points[signed_distances < 0]
    total_points = len(points)
    
    is_upside_down = (len(points_below) / total_points) > 0.2
    
    orientation_info = {
        "points_above_floor": points_above,
        "points_below_floor": points_below,
        "ratio_above": points_above / total_points,
        "ratio_below": points_below / total_points,
        "is_upside_down": is_upside_down,
        "floor_normal": normal_vector
    }
    
    return orientation_info

def align_to_xy_plane(points, plane_model, orientation_info):
    """Align the point cloud so the floor is parallel to the XY plane and positioned at z=0."""
    print("Aligning point cloud to XY plane...")
    # Extract plane parameters
    a, b, c, d = plane_model
    floor_normal = np.array([a, b, c])
    
    # If the model is upside down, flip the normal
    if orientation_info["is_upside_down"]:
        floor_normal = -floor_normal
    
    # Define the target normal (Z-axis)
    z_axis = np.array([0, 0, 1])
    
    # Calculate rotation required to align floor_normal with Z-axis
    rotation_axis = np.cross(floor_normal, z_axis)
    norm_rotation_axis = np.linalg.norm(rotation_axis)
    
    if norm_rotation_axis < 1e-6:
        # The normals are already aligned or opposite
        if np.dot(floor_normal, z_axis) < 0:
            rotation_matrix = -np.eye(3)
        else:
            rotation_matrix = np.eye(3)
    else:
        rotation_axis /= norm_rotation_axis
        rotation_angle = np.arccos(np.clip(np.dot(floor_normal, z_axis), -1.0, 1.0))
        rotation = Rotation.from_rotvec(rotation_angle * rotation_axis)
        rotation_matrix = rotation.as_matrix()
    
    # Rotate all points
    rotated_points = (rotation_matrix @ points.T).T
    
    # Find a point on the original plane
    plane_norm_sq = a**2 + b**2 + c**2
    if plane_norm_sq == 0:
        raise ValueError("Invalid plane model with zero normal vector.")
    p0 = np.array([-a * d / plane_norm_sq,
                   -b * d / plane_norm_sq,
                   -c * d / plane_norm_sq])
    
    # Rotate the point on the plane
    p0_rotated = rotation_matrix @ p0
    
    # Calculate translation to bring the rotated plane to z=0
    translation_z = -p0_rotated[2]
    translation = np.array([0, 0, translation_z])
    
    # Apply translation
    aligned_points = rotated_points + translation
    
    return aligned_points, rotation_matrix, translation


def run_floor_separation(points, out_path, min_floor_points=500, distance_threshold=None):
    """Complete pipeline to process the point cloud."""
    result = {}
    # 1. Find floor
    
    if distance_threshold is None:
        print("Floor separation: Searching for optimal threshold...")
        floor_points, non_floor_points, plane_model = find_floor_plane_auto(
            points, 
            out_path,
            min_floor_points=min_floor_points
        )
    else:
        # floor_points, non_floor_points, plane_model = find_largest_surface_floor(points, distance_threshold=distance_threshold, 
        #                   min_floor_points=min_floor_points)
        print(f"Floor separation: Using fixed threshold {distance_threshold:.4f}")
        floor_points, non_floor_points, plane_model = find_floor_plane(points, distance_threshold=distance_threshold, 
                          min_floor_points=min_floor_points)
    result['floor_points'] = floor_points
    result['non_floor_points'] = non_floor_points
    result['plane_model'] = plane_model
    
    # 2. Determine model orientation
    orientation_info = determine_model_orientation(
        non_floor_points, 
        plane_model
    )
    
    # 3. Align to XY plane
    aligned_points, rotation_matrix, translation = align_to_xy_plane(
        non_floor_points, 
        plane_model, 
        orientation_info
    )
    result['aligned_points'] = aligned_points
    result['orientation_info'] = orientation_info
    result['transformation'] = {
        'rotation': rotation_matrix,
        'translation': translation
    }

    final_points = orientation_info["points_below_floor"] if orientation_info["is_upside_down"] else orientation_info["points_above_floor"]
    result['discarded_points'] = orientation_info["points_above_floor"] if orientation_info["is_upside_down"] else orientation_info["points_below_floor"]

    
    # 4. Remove points below floor
    # final_points = remove_points_below_floor(final_points, plane_model)
    result['final_points'] = final_points

    
    return result