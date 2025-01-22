from scipy.spatial import ConvexHull
from tqdm import tqdm

from colmap_loader import read_extrinsics_binary, read_intrinsics_binary

from utils import read_ply, write_ply
from clustering import get_densest_cluster, remove_outliers
from floor import run_floor_separation
from project import project_points_to_image, poisson_surface_reconstruction




def process_object(object_name, input_path, output_path, plot=False, kde_samples=1000, min_peak_points=300, project=False, outlier_removal_eps=0.4, separate_floor=True, poisson_depth=9):
    print("start")
    points_original, colors, normals = read_ply(f'{input_path}/{object_name}/sparse/0/points3D.ply')
    object_points, density = get_densest_cluster(points_original, min_peak_points, kde_samples=kde_samples, sigma=1, plot=plot)
    write_ply(points_original, object_points, colors, normals, f"{output_path}/{object_name}/new_{object_name}_dense.ply")
    if separate_floor:
        result = run_floor_separation(object_points, distance_threshold=0.3)
        write_ply(points_original, result["final_points"], colors, normals, f"{output_path}/{object_name}/new_{object_name}_floorseg.ply")
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
    points_path = f"{output_path}/{object_name}/new_{object_name}_final.ply"
    
    write_ply(points_original, outliers_removed, colors, normals, points_path)

    result["mesh"] = mesh
    if project:
        print("Projecting points to images...")
        hull = ConvexHull(outliers_removed)
        result["hull"] = hull
        for camera_id in tqdm(cam_extrinsics, desc="Projecting Cameras"):
            project_points_to_image(
                mesh, hull, camera_id, cam_extrinsics, cam_intrinsics,
                object_name, outliers_removed, input_path, output_path
            )
    return result
    

process_object("kitchen", "/shared/home/nlr/dang/data/mipnerf/", "/shared/home/nlr/dang/output/mipnerf/", project=True, separate_floor=True)