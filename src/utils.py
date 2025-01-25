from plyfile import PlyData
import numpy as np
import open3d as o3d
import copy
from colmap_loader import qvec2rotmat



def read_ply(file_path):
    print(f"Reading {file_path}")
    plydata = PlyData.read(file_path)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    if 'red' in plydata['vertex']:
        r = plydata['vertex']['red']
        g = plydata['vertex']['green']
        b = plydata['vertex']['blue']
        colors = np.array([r, g, b]).T
    else:
        colors = None
    if 'nx' in plydata['vertex']:
        nx = plydata['vertex']['nx']
        ny = plydata['vertex']['ny']
        nz = plydata['vertex']['nz']
        normals = np.array([nx, ny, nz]).T
    else:
        normals = None
    points = np.array([x, y, z]).T
    return points, colors, normals


def get_extrinsic_matrix(qvec, tvec):
    """
    Create a 4x4 extrinsic matrix from quaternion and translation vector.
    
    Parameters:
    - qvec: np.ndarray - Quaternion (w, x, y, z) representing rotation.
    - tvec: np.ndarray - Translation vector (x, y, z).
    
    Returns:
    - extrinsic_matrix: np.ndarray - 4x4 extrinsic matrix.
    """
    # Convert quaternion to rotation matrix
    rotation_matrix = qvec2rotmat(qvec)
    
    # Create 4x4 extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = tvec
    
    return extrinsic_matrix

def rotate_ply(ply, output_file):
    output_file = output_file.replace(".ply", "_flipped.ply")
    new_ply = copy.deepcopy(ply)
    rotation_matrix = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    if isinstance(ply, o3d.geometry.TriangleMesh):
        vertices = np.asarray(ply.vertices)
    else:
        vertices = np.asarray(ply.points)
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    if isinstance(ply, o3d.geometry.TriangleMesh):
        new_ply.vertices = o3d.utility.Vector3dVector(rotated_vertices)
        o3d.io.write_triangle_mesh(output_file, new_ply)
    else:
        new_ply.points = o3d.utility.Vector3dVector(rotated_vertices)
        o3d.io.write_point_cloud(output_file, new_ply)


def write_pc(original_points, points, colors, normals, out_path, special_points=None):
    dtype = [('x', float), ('y', float), ('z', float)]
    original_structured = np.array([tuple(p) for p in original_points], dtype=dtype)
    outliers_structured = np.array([tuple(p) for p in points], dtype=dtype)

    indices = np.nonzero(np.isin(original_structured, outliers_structured))[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        end_colors = colors[indices]
        final_colors = end_colors.copy()
        if special_points is not None:
            special_indices = np.nonzero(np.isin(points, special_points))[0]
            final_colors[special_indices] = [255, 0, 0]
        pcd.colors = o3d.utility.Vector3dVector(final_colors / 255.0)
    if normals is not None:
        end_normals = normals[indices]
        pcd.normals = o3d.utility.Vector3dVector(end_normals)
    o3d.io.write_point_cloud(out_path, pcd)
    rotate_ply(pcd, out_path)
    return pcd


def write_mesh(mesh, out_path):
    o3d.io.write_triangle_mesh(out_path, mesh)
    rotate_ply(mesh, out_path)
    return mesh