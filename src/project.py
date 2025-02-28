from pathlib import Path
import os
import numpy as np
import open3d as o3d
import cv2
import time

from utils import get_extrinsic_matrix
from scipy.spatial import Delaunay

def delaunay_surface_reconstruction(points):
    """
    Perform surface reconstruction using Delaunay triangulation.
    
    Parameters:
    -----------
    points : np.ndarray
        Input point cloud as a NumPy array of shape (N, 3).

    Returns:
    --------
    mesh : o3d.geometry.TriangleMesh
        The reconstructed mesh using Delaunay triangulation.
    """
    print("Running Delaunay surface reconstruction...")

    # Ensure points are in 3D space
    if points.shape[1] != 3:
        raise ValueError("Input points must be a 3D array of shape (N, 3).")

    # Perform Delaunay triangulation in 2D (projected to XY plane)
    points_2d = points[:, :2]  # Use XY coordinates for 2D triangulation
    delaunay = Delaunay(points_2d)

    # Create the Open3D mesh from Delaunay simplices (triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(delaunay.simplices)

    # Optionally remove low-density vertices
    # densities = mesh.compute_vertex_normals()  # Compute normals to estimate densities
    # vertices_to_remove = densities < np.quantile(densities, 0.05)
    # mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh

def project_mesh(mesh, camera_intrinsics, extrinsics):
    """Project 3D mesh vertices onto a 2D image plane."""
    vertices = np.asarray(mesh.vertices)
    vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    
    if extrinsics.shape != (4, 4):
        raise ValueError("Extrinsics must be a 4x4 matrix.")
    
    # Transform vertices to camera space
    vertices_camera = (extrinsics @ vertices_homogeneous.T).T
    vertices_camera = vertices_camera[:, :3]

    # Apply mask to keep only vertices with positive Z
    mask = vertices_camera[:, 2] > 0
    vertices_camera = vertices_camera[mask]

    # Update faces to match the filtered vertices
    valid_indices = np.where(mask)[0]
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
    faces = np.asarray(mesh.triangles)

    # Remap and filter faces
    remapped_faces = []
    for face in faces:
        if all(v in index_mapping for v in face):  # Keep faces with all vertices valid
            remapped_faces.append([index_mapping[v] for v in face])
    faces = np.array(remapped_faces)  # Convert the valid faces to a NumPy array

    # Project vertices onto the 2D plane
    fx, fy, cx, cy = camera_intrinsics.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    projected_vertices = (K @ vertices_camera.T).T
    projected_vertices = projected_vertices[:, :2] / projected_vertices[:, 2].reshape(-1, 1)

    return projected_vertices.astype(int), mask, faces


def render_mesh_image(mesh, camera, extrinsics, image_path, output_path, mask_path, image_size=None, fill_color=(0, 255, 0), alpha=0.5):
    """Render a composite image with the projected mesh and a masked original image."""
    if image_size is None:
        image_size = (camera.height, camera.width)
    projected_vertices, mask, faces = project_mesh(mesh, camera, extrinsics)
    mesh_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    mask_image = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)  # Mask for the mesh

    for face in faces:
        face_vertices = projected_vertices[face]
        cv2.fillConvexPoly(mask_image, face_vertices, 255)  # Fill the face on the mask
        cv2.fillConvexPoly(mesh_image, face_vertices, fill_color)  # Optionally fill for visualization

    # closing the mask
    # kernel = np.ones((50,50),np.uint8)
    # mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(mask_path, mask_image)

    # 2. Load the actual image and create an overlay (middle side)
    actual_image = cv2.imread(image_path)
    if actual_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if actual_image.shape[:2] != image_size:
        actual_image = cv2.resize(actual_image, (image_size[1], image_size[0]))

    overlay_image = cv2.addWeighted(mesh_image, alpha, actual_image, 1 - alpha, 0)

    # 3. Create the masked original image using the mesh mask
    masked_image = cv2.bitwise_and(actual_image, actual_image, mask=mask_image)

    # 4. Concatenate the four images side-by-side, including the masked original as the fourth
    composite_image = np.hstack((mesh_image, overlay_image, actual_image, masked_image))

    # Resize the composite image if it's too large
    composite_image = cv2.resize(composite_image, (composite_image.shape[1]//4, composite_image.shape[0]//4))
    cv2.imwrite(output_path, composite_image)
    
    return composite_image


def project_points_to_image(mesh, camera_id, cam_extrinsics, cam_intrinsics, object_name, input_path, output_base_path, mesh_type):
    camera = cam_intrinsics[camera_id] if camera_id in cam_intrinsics else cam_intrinsics[list(cam_intrinsics.keys())[0]]
    qvec = cam_extrinsics[camera_id].qvec
    tvec = cam_extrinsics[camera_id].tvec
    extrinsics = get_extrinsic_matrix(qvec, tvec)
    name = cam_extrinsics[camera_id].name
    # "/shared/home/nlr/dang/data/mipnerf/", "/shared/home/nlr/dang/output/mipnerf/"
    image_path = f"{input_path}/{object_name}/images/{name}"  # Path to the image in images_8
    output_name = Path(image_path).stem

    output_dir = f"{output_base_path}/{mesh_type}/mask"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_base_path}/{mesh_type}/{output_name}.png"
    mask_path = f"{output_base_path}/{mesh_type}/mask/{output_name}.png"
    render_mesh_image(mesh, camera, extrinsics, image_path, output_path, mask_path)


def poisson_surface_reconstruction(points, depth=None):
    print("Running Poisson surface reconstruction...")
    if depth is None:
        return optimize_poisson_depth(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=100))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return mesh


def optimize_poisson_depth(
    points, 
    depth_range=(5, 12), 
    metric='surface_area'
):
    depths = list(range(depth_range[0], depth_range[1] + 1))
    start_time = time.perf_counter()
    results = []
    for depth in depths:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(k=30)
            
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
            
            vertices_to_remove = densities < np.quantile(densities, 0.05)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            results.append({
                'depth': depth,
                'surface_area': mesh.get_surface_area(),
                'num_triangles': len(mesh.triangles),
                'num_vertices': len(mesh.vertices),
                'mesh': mesh	
            })
        except Exception as e:
            print(f"Error at depth {depth}: {e}")
    
    if metric == 'surface_area':
        best_result = max(results, key=lambda x: x['surface_area'])
    else:
        best_result = results[len(results) // 2]
    
    print(f"Poisson surface reconstruction took {time.perf_counter() - start_time:.2f} seconds. Best depth: {best_result['depth']}")
    return best_result['mesh']