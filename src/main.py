from tqdm import tqdm
from datetime import datetime
import os
import numpy as np

from colmap_loader import read_extrinsics_binary, read_intrinsics_binary

from utils import read_ply, write_mesh, write_pc
from clustering import get_densest_cluster, remove_outliers, denoise_point_cloud
from floor import run_floor_separation
from project import project_points_to_image, poisson_surface_reconstruction, delaunay_surface_reconstruction
from metrics import ply_metrics, mask_metrics
import signal
import sys

def setup_signal_handler():
    """
    Sets up a signal handler to gracefully exit on Ctrl+C.
    Stops all running threads/processes and exits cleanly.
    """
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Exiting gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

setup_signal_handler()


NOW_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def process_object(object_name, input_path, output_path, kde_samples=1000, min_peak_points=300, double_dbscan=True, project=False, outlier_removal_eps=0.4, separate_floor=True, floor_distance_threshold=None, poisson_depth=9, point_cloud_metrics=True):
    print("start")
    output_path = f"{output_path}/{object_name}/{NOW_STR}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plot_path = f"{output_path}/plots"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    points_original, colors, normals = read_ply(f'{input_path}/{object_name}/sparse/0/points3D.ply')
    object_points, density = get_densest_cluster(points_original, output_path, min_peak_points, kde_samples=kde_samples, sigma=1)
    if separate_floor:
        result = run_floor_separation(object_points, output_path, distance_threshold=floor_distance_threshold)
        write_pc(points_original, result["final_points"], colors, normals, f"{output_path}/{object_name}_floorseg.ply")
    else:
        result = {"final_points": object_points}
    write_pc(points_original, object_points, colors, normals, f"{output_path}/{object_name}_dense.ply", special_points=np.concatenate((result["discarded_points"], result["floor_points"]), axis=0))
    result["densest_cluster"] = object_points
    result["density"] = density
    outliers_removed = remove_outliers(result["final_points"], eps=outlier_removal_eps, min_samples=50)
    points_path = f"{output_path}/{object_name}_stat_outlier.ply"
    stat_outlier = denoise_point_cloud(result["final_points"], neighbors=200, std_ratio=1.5)
    write_pc(points_original, stat_outlier, colors, normals, points_path)
    if double_dbscan:
        outliers_removed = remove_outliers(outliers_removed, eps=outlier_removal_eps, min_samples=200)
    result["outliers_removed"] = outliers_removed
    if object_name[-1].isdigit():
        object_name = object_name[:object_name.rfind("_")]
    cam_extrinsics = read_extrinsics_binary(f"{input_path}/{object_name}/sparse/0/images.bin")
    cam_intrinsics = read_intrinsics_binary(f"{input_path}/{object_name}/sparse/0/cameras.bin")
    
    
    # Write point cloud to file
    points_path = f"{output_path}/{object_name}_final.ply"
    
    write_pc(points_original, outliers_removed, colors, normals, points_path)
    if point_cloud_metrics:
        ply_metrics(f"{input_path}/{object_name}/points3D_gt.ply", output_path, object_name, f"{output_path}/metrics_ply.json")


    mesh = poisson_surface_reconstruction(outliers_removed, depth=poisson_depth)
    write_mesh(mesh, f"{output_path}/{object_name}_mesh.ply")

    delaunay = delaunay_surface_reconstruction(outliers_removed)
    write_mesh(delaunay, f"{output_path}/{object_name}_delaunay.ply")

    result["mesh"] = mesh
    if project:
        print("Projecting points to images...")
        mesh_type = "mesh"
        for camera_id in tqdm(cam_extrinsics, desc="Projecting Cameras"):
            project_points_to_image(
                mesh, camera_id, cam_extrinsics, cam_intrinsics,
                object_name, input_path, output_path, mesh_type
            )
        mask_metrics(f"{input_path}/{object_name}/images", f"{output_path}/{mesh_type}/mask", f"{output_path}/metrics_mask.json")
    return result
    

# process_object("stump", "S:/git/sfm-ooi-extract/data", "S:/git/sfm-ooi-extract/output", double_dbscan=False, project=True, point_cloud_metrics=False, separate_floor=True, min_peak_points=200) # [0][0]
process_object("kitchen", "S:/git/sfm-ooi-extract/data", "S:/git/sfm-ooi-extract/output", project=False) # [0][0]
# process_object("bonsai", "S:/git/sfm-ooi-extract/data", "S:/git/sfm-ooi-extract/output", project=False) # [0][0]
# process_object("bicycle", "S:/git/sfm-ooi-extract/data", "S:/git/sfm-ooi-extract/output", double_dbscan=False, project=True, outlier_removal_eps=0.7, floor_distance_threshold=0.08, poisson_depth=10) # [1][1]
# process_object("garden", "S:/git/sfm-ooi-extract/data", "S:/git/sfm-ooi-extract/output", outlier_removal_eps=1, floor_distance_threshold=0.08, double_dbscan=False, project=False) # [1][1]