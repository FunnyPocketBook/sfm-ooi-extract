import numpy as np
from scipy.spatial import ConvexHull

import open3d as o3d

from colmap_loader import read_extrinsics_binary, read_intrinsics_binary

from utils import read_ply
from clustering import get_densest_cluster, remove_outliers
from floor import run_floor_separation
from project import project_points_to_image, poisson_surface_reconstruction




def process_object(object_name, input_path, output_path, plot=False, kde_samples=1000, min_peak_points=300, project=False, outlier_removal_eps=0.4, separate_floor=True, poisson_depth=9):
    points_original, colors, normals = read_ply(f'{input_path}/points_{object_name}.ply')
    # plot_point_cloud(bonsai_points, bonsai_colors)
    object_points, density = get_densest_cluster(points_original, min_peak_points, kde_samples=kde_samples, sigma=1, plot=plot)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_points)
    o3d.io.write_point_cloud("{output_path}/{object_name}/cluster/new_{object_name}_density.ply", pcd)
    if separate_floor:
        result = run_floor_separation(object_points, distance_threshold=0.3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(result["final_points"])
        o3d.io.write_point_cloud("{output_path}/{object_name}/cluster/new_{object_name}_floorseg.ply", pcd)
    else:
        result = {"final_points": object_points}
    result["densest_cluster"] = object_points
    result["density"] = density
    outliers_removed = remove_outliers(result["final_points"], eps=outlier_removal_eps, min_samples=50)
    outliers_removed = remove_outliers(outliers_removed, eps=outlier_removal_eps, min_samples=200)
    result["outliers_removed"] = outliers_removed
    if object_name[-1].isdigit():
        object_name = object_name[:object_name.rfind("_")]
    cam_extrinsics = read_extrinsics_binary(f"{input_path}/{object_name}/sparse/0/images.bin")
    cam_intrinsics = read_intrinsics_binary(f"{input_path}/{object_name}/sparse/0/cameras.bin")
    
    mesh = poisson_surface_reconstruction(outliers_removed, depth=poisson_depth)
    
    # Write point cloud to file
    points_path = f"{output_path}/{object_name}/cluster/new_{object_name}_final.ply"
    
    dtype = [('x', float), ('y', float), ('z', float)]
    original_structured = np.array([tuple(p) for p in points_original], dtype=dtype)
    outliers_structured = np.array([tuple(p) for p in outliers_removed], dtype=dtype)

    indices = np.nonzero(np.in1d(original_structured, outliers_structured))[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(outliers_removed)
    if colors is not None:
        end_colors = colors[indices]
        pcd.colors = o3d.utility.Vector3dVector(end_colors / 255.0)
    if normals is not None:
        end_normals = normals[indices]
        pcd.normals = o3d.utility.Vector3dVector(end_normals)

    o3d.io.write_point_cloud(points_path, pcd)

    result["mesh"] = mesh
    if project:
        hull = ConvexHull(outliers_removed)
        result["hull"] = hull
        for camera_id in cam_extrinsics:
            project_points_to_image(mesh, hull, camera_id, cam_extrinsics, cam_intrinsics, object_name, outliers_removed)
    return result
    

